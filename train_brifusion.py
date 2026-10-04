"""Two-stage BRIFusion training with explicit data, device, and checkpoint settings."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from dataloader.brifusion_data import FusionDataset
from model.brifusion import BRIFusion
from utils.bri_losses import BRIFusionLoss, LossConfig
from utils.degradations import compound_degrade, local_intervention


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('format') != 'BRIFusion-v1':
        raise ValueError('Expected a BRIFusion-v1 checkpoint. Legacy DAMFusion weights are not architecture-compatible; retrain stage 0.')
    return checkpoint


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def prepare_batch(batch, device, mode):
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    if mode == 'synthetic':
        batch['vi'] = compound_degrade(batch['target_vi'], 0)
        batch['ir'] = compound_degrade(batch['target_ir'], 1)
    return batch


def training_loss(model, criterion, batch, stage, use_intervention):
    output = model(batch['vi'], batch['ir'], return_aux=True, stage=stage)
    intervened = None
    if use_intervention:
        vi, ir, mv, mi = local_intervention(batch['vi'], batch['ir'])
        other = model(vi, ir, return_aux=True, stage=stage)
        intervened = (other, vi, ir, mv, mi)
    return criterion(output, batch, stage, intervened)


def parser(default_stage='fusion'):
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--data', required=True)
    result.add_argument('--val-data')
    result.add_argument('--medical', action='store_true', help='Use CT/MRI folder names for --data and --val-data')
    result.add_argument('--medical-data', action='append', default=[], help='Additional CT/MRI training roots; may be repeated')
    result.add_argument('--mode', choices=['paired', 'synthetic'], default='paired')
    result.add_argument('--stage', choices=['restore', 'fusion'], default=default_stage)
    result.add_argument('--config', default=str(Path(__file__).parent / 'configs/brifusion.json'))
    result.add_argument('--output', default='runs/brifusion')
    result.add_argument('--init', help='Warm-start a new stage using a BRIFusion checkpoint')
    result.add_argument('--resume', help='Resume this stage, including optimizer/scheduler/RNG state')
    result.add_argument('--from-scratch', action='store_true', help='Explicitly allow fusion-stage training without stage-0 weights')
    result.add_argument('--device', default='auto')
    result.add_argument('--epochs', type=int, default=100)
    result.add_argument('--batch-size', type=int, default=4)
    result.add_argument('--crop-size', type=int, default=256)
    result.add_argument('--workers', type=int, default=0)
    result.add_argument('--lr', type=float, default=1e-4)
    result.add_argument('--encoder-lr-scale', type=float, default=0.25)
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--steps-per-epoch', type=int, default=0, help='0 uses the full dataset; positive values are for debugging')
    result.add_argument('--threads', type=int, default=0)
    result.add_argument('--amp', action='store_true')
    result.add_argument('--width', type=int)
    result.add_argument('--no-response', action='store_true')
    result.add_argument('--no-reliability', action='store_true')
    result.add_argument('--no-intervention', action='store_true')
    return result


def main(argv=None, default_stage='fusion'):
    args = parser(default_stage).parse_args(argv)
    if args.init and args.resume:
        raise ValueError('Use either --init or --resume, not both')
    if args.stage == 'fusion' and not (args.init or args.resume or args.from_scratch):
        raise ValueError('Fusion training needs --init STAGE0/last.pth, or explicit --from-scratch')
    if args.epochs < 1 or args.batch_size < 1 or args.crop_size < 0 or args.steps_per_epoch < 0:
        raise ValueError('Invalid training sizes or epoch count')
    if args.threads:
        torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if args.amp and device.type != 'cuda':
        raise ValueError('--amp requires a CUDA device')
    settings = json.loads(Path(args.config).read_text(encoding='utf-8'))
    checkpoint = load_checkpoint(args.resume or args.init) if (args.resume or args.init) else None
    model_config = dict(checkpoint['model_config'] if checkpoint else settings['model'])
    if args.width is not None:
        model_config['width'] = args.width
    if args.no_response:
        model_config['use_response'] = False
    if args.no_reliability:
        model_config['use_reliability'] = False
    if checkpoint and model_config != checkpoint['model_config']:
        raise ValueError('Checkpoint architecture must match. Train each structural ablation from stage 0.')
    model = BRIFusion(**model_config).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    # The default executable path uses image-derived conditioning, without CLIP weights.
    model.external_condition.requires_grad_(False)
    if args.stage == 'restore':
        model.fusion_gata.requires_grad_(False)
        model.decode_fi.requires_grad_(False)
    feature_parameters, remaining_parameters = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = feature_parameters if name.startswith(('conditioners.', 'stem.', 'encoder_moe.', 'down.', 'enhancers.')) else remaining_parameters
            group.append(parameter)
    feature_lr = args.lr * (args.encoder_lr_scale if args.stage == 'fusion' else 1)
    optimizer = torch.optim.AdamW([{'params': feature_parameters, 'lr': feature_lr},
                                  {'params': remaining_parameters, 'lr': args.lr}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_config = LossConfig(**(checkpoint['loss_config'] if args.resume else settings.get('loss', {})))
    criterion = BRIFusionLoss(loss_config, model.use_reliability)
    use_intervention = checkpoint['use_intervention'] if args.resume else not args.no_intervention
    if args.resume and args.no_intervention and use_intervention:
        raise ValueError('Resume preserves intervention settings; use --init to start a different training run')
    datasets = [FusionDataset(args.data, args.mode, args.crop_size, medical=args.medical)]
    datasets += [FusionDataset(path, args.mode, args.crop_size, medical=True) for path in args.medical_data]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        worker_init_fn=seed_worker, generator=generator, pin_memory=device.type == 'cuda')
    validation = None
    if args.val_data:
        validation = DataLoader(FusionDataset(args.val_data, args.mode, args.crop_size, augment=False, medical=args.medical), batch_size=1)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    start_epoch, global_step, best = 0, 0, float('inf')
    if args.resume:
        if checkpoint['stage'] != args.stage:
            raise ValueError('Resume must use the saved stage; use --init for restoration-to-fusion transfer')
        for key in ('mode', 'crop_size', 'batch_size', 'epochs'):
            if checkpoint['args'][key] != getattr(args, key):
                raise ValueError(f'Resume requires matching --{key.replace("_", "-")}; use --init for a new schedule')
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch, global_step, best = checkpoint['epoch'] + 1, checkpoint['step'], checkpoint['best_validation']
        torch.set_rng_state(checkpoint['torch_rng'])
        generator.set_state(checkpoint['loader_rng'])
        if device.type == 'cuda' and checkpoint.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    if start_epoch >= args.epochs:
        raise ValueError('Checkpoint has already completed the requested epochs; use --init for a new training schedule')
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'config.json').write_text(json.dumps(dict(args=vars(args), model=model.config, loss=asdict(loss_config), use_intervention=use_intervention), indent=2), encoding='utf-8')
    for epoch in range(start_epoch, args.epochs):
        model.train()
        sums, count = {}, 0
        for batch_index, batch in enumerate(loader):
            if args.steps_per_epoch and batch_index >= args.steps_per_epoch:
                break
            batch = prepare_batch(batch, device, args.mode)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp):
                loss, components = training_loss(model, criterion, batch, args.stage, use_intervention)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            count += 1
            values = {name: float(value.detach()) for name, value in components.items()}
            values['total'] = float(loss.detach())
            for name, value in values.items():
                sums[name] = sums.get(name, 0) + value
            if count == 1 or count % 10 == 0:
                print(json.dumps(dict(epoch=epoch + 1, step=global_step, **values)), flush=True)
        scheduler.step()
        metrics = dict(epoch=epoch + 1, step=global_step, train={name: value/count for name, value in sums.items()})
        improved = False
        if validation is not None:
            model.eval()
            val_losses = []
            # Fixed validation corruption while preserving the training RNG sequence.
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []), torch.no_grad():
                torch.manual_seed(args.seed + 10000)
                for batch in validation:
                    batch = prepare_batch(batch, device, args.mode)
                    value, _ = training_loss(model, criterion, batch, args.stage, False)
                    val_losses.append(float(value))
            metrics['validation_loss'] = sum(val_losses) / len(val_losses)
            improved = metrics['validation_loss'] < best
            best = min(best, metrics['validation_loss'])
        with (destination / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(metrics) + '\n')
        state = dict(format='BRIFusion-v1', model_config=model.config, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                     loss_config=asdict(loss_config), stage=args.stage, epoch=epoch, step=global_step,
                     use_intervention=use_intervention, best_validation=best, args=vars(args),
                     torch_rng=torch.get_rng_state(), loader_rng=generator.get_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
        temporary = destination / 'last.tmp'
        torch.save(state, temporary)
        temporary.replace(destination / 'last.pth')
        if improved:
            torch.save(state, destination / 'best.pth')
        print(json.dumps(metrics), flush=True)
    return destination / 'last.pth'


if __name__ == '__main__':
    main()
