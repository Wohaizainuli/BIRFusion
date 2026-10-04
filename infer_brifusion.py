"""Paired inference, reliability visualization, and model-only latency measurement."""
import argparse
import json
from pathlib import Path
import time
import torch
from dataloader.brifusion_data import image_index, load_image, luminance, colorize, save_image
from model.brifusion import BRIFusion
from train_brifusion import load_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--vi', required=True, help='Visible/CT image directory')
    parser.add_argument('--ir', required=True, help='Infrared/MRI image directory')
    parser.add_argument('--output', default='results/brifusion')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--save-quality', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--threads', type=int, default=0)
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint['stage'] != 'fusion':
        raise ValueError('Inference requires a fusion-stage checkpoint; the stage-0 fusion decoder is untrained')
    model = BRIFusion(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    visible, infrared = image_index(args.vi), image_index(args.ir)
    if set(visible) != set(infrared):
        raise ValueError('Visible and infrared image stems do not match')
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    synchronize = lambda: torch.cuda.synchronize(device) if device.type == 'cuda' else None
    timings, shapes = [], []
    with torch.inference_mode():
        for index, key in enumerate(sorted(visible)):
            rgb = load_image(visible[key]).to(device)
            vi, ir = luminance(rgb).unsqueeze(0), luminance(load_image(infrared[key])).unsqueeze(0).to(device)
            if vi.shape != ir.shape:
                raise ValueError(f'Pair is not aligned in size: {key}')
            if index == 0:
                for _ in range(args.warmup):
                    model(vi, ir, return_aux=True)
            synchronize()
            start = time.perf_counter()
            output = model(vi, ir, return_aux=True)
            synchronize()
            timings.append(time.perf_counter() - start)
            shapes.append(list(vi.shape[-2:]))
            fused = output.fused[0] if args.grayscale else colorize(output.fused[0], rgb)
            save_image(fused, destination / (visible[key].stem + '.png'))
            if args.save_quality:
                quality_dir = destination / 'quality'
                quality_dir.mkdir(exist_ok=True)
                for suffix, values in (('vi', output.reliability_vi), ('ir', output.reliability_ir)):
                    save_image(torch.stack(values).mean(0)[0], quality_dir / f'{visible[key].stem}_{suffix}.png')
    report = dict(parameters=sum(p.numel() for p in model.parameters()), images=len(timings),
                  input_shapes=shapes, model_seconds=sum(timings), mean_latency_ms=sum(timings)/len(timings)*1000,
                  fps=len(timings)/sum(timings), device=str(device), torch_version=str(torch.__version__),
                  timing_scope='Full model forward including image conditioners and restoration heads; excludes I/O and device transfer')
    (destination / 'timing.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    main()
