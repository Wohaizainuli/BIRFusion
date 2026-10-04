import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from model.brifusion import BRIFusion
from model.bri_layers import BioInspiredAdaptiveResponseEnhancer, ReliabilityGuidedFusion, window_partition, window_merge
from dataloader.brifusion_data import FusionDataset
from utils.bri_losses import (BRIFusionLoss, LossConfig, reliability_target,
                              reliability_ranking, intervention_consistency, ssim_loss)
from utils.degradations import local_intervention
from train_brifusion import training_loss, main as train, load_checkpoint
from infer_brifusion import main as infer


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(123)


def small_model(**kwargs):
    return BRIFusion(width=4, condition_channels=4, **kwargs)


@pytest.mark.parametrize('shape', [(1, 1, 16, 16), (2, 1, 33, 35), (1, 1, 1, 1)])
def test_forward_size_range_and_finiteness(shape):
    model = small_model().eval()
    vi, ir = torch.rand(shape), torch.rand(shape)
    with torch.no_grad():
        output = model(vi, ir, return_aux=True)
    for value in (output.fused, output.restored_vi, output.restored_ir, *output.reliability_vi):
        assert value.shape == vi.shape
        assert torch.isfinite(value).all()
        assert value.min() >= 0 and value.max() <= 1
    for route in output.routes:
        torch.testing.assert_close(route.sum(1), torch.ones_like(route[:, 0]))
        assert ((route > 0).sum(1) == 2).all()


def test_batch_independent_deterministic_evaluation():
    model = small_model().eval()
    vi, ir = torch.rand(2, 1, 32, 32), torch.rand(2, 1, 32, 32)
    with torch.no_grad():
        single = model(vi[:1], ir[:1])[2]
        together = model(vi, ir)[2][:1]
        again = model(vi[:1], ir[:1])[2]
    torch.testing.assert_close(single, together, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(single, again, atol=0, rtol=0)


def test_response_identity_option_and_extreme_input_gradients():
    module = BioInspiredAdaptiveResponseEnhancer(8, condition_channels=4, initial_gain=0)
    x = torch.randn(2, 8, 9, 11, requires_grad=True)
    torch.testing.assert_close(module(x), x)
    module.residual_gain.data.fill_(0.05)
    x = (torch.randn(2, 8, 9, 11) * 100).requires_grad_()
    result = module(x, torch.rand(2, 4, 9, 11))
    result.square().mean().backward()
    assert torch.isfinite(result).all() and torch.isfinite(x.grad).all()
    assert torch.isfinite(module.response_order.grad).all()


def test_window_roundtrip_odd_geometry():
    x = torch.rand(2, 3, 7, 9)
    tiles, info = window_partition(x, 4)
    torch.testing.assert_close(window_merge(tiles, info), x)


def test_reliability_controls_modality_weighting():
    layer = ReliabilityGuidedFusion(8)
    torch.nn.init.zeros_(layer.content.weight)
    torch.nn.init.zeros_(layer.content.bias)
    layer.residual_gain.data.zero_()
    vi, ir = torch.ones(1, 8, 8, 8), torch.zeros(1, 8, 8, 8)
    qv, qi = torch.full((1, 1, 8, 8), 0.1), torch.full((1, 1, 8, 8), 0.9)
    output, _, _, weights = layer(vi, ir, qv, qi)
    torch.testing.assert_close(output, torch.full_like(output, 0.1))
    torch.testing.assert_close(weights[:, :1], qv)


def test_sparse_dispatch_skips_unselected_expert_and_trains_router():
    layer = ReliabilityGuidedFusion(8)
    torch.nn.init.zeros_(layer.router[-1].weight)
    layer.router[-1].bias.data.copy_(torch.tensor([3.0, 2.0, -9.0]))
    calls = [0, 0, 0]
    handles = []
    for index, expert in enumerate(layer.experts):
        def hook(module, inputs, result, index=index):
            calls[index] += inputs[0].shape[0]
        handles.append(expert.register_forward_hook(hook))
    vi, ir = torch.rand(2, 8, 8, 8), torch.rand(2, 8, 8, 8)
    q = torch.ones(2, 1, 8, 8)
    output, balance, _, _ = layer(vi, ir, q, q)
    (output.square().mean() + balance * 0.01).backward()
    assert calls == [8, 8, 0]
    assert layer.router[-1].bias.grad.abs().sum() > 0
    for handle in handles:
        handle.remove()


def test_reliability_target_is_reference_discrepancy():
    reference = torch.full((1, 1, 16, 16), 0.5)
    clean = reliability_target(reference, reference)
    degraded = reliability_target(reference * 0.2, reference)
    assert torch.all(clean == 1)
    assert torch.all(degraded < clean)


def test_ranking_sign_and_invalid_interventions():
    before = torch.full((1, 1, 8, 8), 0.9, requires_grad=True)
    bad_after = torch.full_like(before, 0.95, requires_grad=True)
    good_after = torch.full_like(before, 0.5, requires_grad=True)
    ones, lower = torch.ones_like(before), torch.full_like(before, 0.2)
    loss = reliability_ranking(before, bad_after, ones, lower, ones)
    assert loss > 0
    loss.backward()
    assert bad_after.grad.mean() > 0
    assert before.grad is None  # Detached teacher for the intervention comparison.
    assert reliability_ranking(before, good_after, ones, lower, ones) == 0
    assert reliability_ranking(before, bad_after, lower, ones, ones) == 0


def test_consistency_only_penalizes_outside_dilated_mask():
    before = torch.zeros(1, 1, 16, 16)
    mask = torch.zeros_like(before)
    mask[..., 6:10, 6:10] = 1
    changed_inside = before + mask
    assert intervention_consistency(before, changed_inside, mask, radius=2) == 0
    changed_outside = changed_inside.clone()
    changed_outside[..., 0, 0] = 1
    assert intervention_consistency(before, changed_outside, mask, radius=2) > 0
    assert intervention_consistency(before, changed_outside, torch.ones_like(mask)) == 0


def test_intervention_is_local_single_modality_and_preserves_inputs():
    vi, ir = torch.rand(16, 1, 32, 32), torch.rand(16, 1, 32, 32)
    vi_original, ir_original = vi.clone(), ir.clone()
    av, ai, mv, mi = local_intervention(vi, ir)
    assert ((mv.sum((1, 2, 3)) > 0) ^ (mi.sum((1, 2, 3)) > 0)).all()
    assert mv.sum() > 0 and mi.sum() > 0
    assert torch.equal((av-vi)*(1-mv), torch.zeros_like(vi))
    assert torch.equal((ai-ir)*(1-mi), torch.zeros_like(ir))
    torch.testing.assert_close(vi, vi_original)
    torch.testing.assert_close(ir, ir_original)


def test_all_three_innovations_receive_finite_gradients():
    model = small_model()
    criterion = BRIFusionLoss(LossConfig(exclusion_radius=2))
    batch = dict(vi=torch.rand(2, 1, 32, 32), ir=torch.rand(2, 1, 32, 32),
                 target_vi=torch.rand(2, 1, 32, 32), target_ir=torch.rand(2, 1, 32, 32))
    loss, values = training_loss(model, criterion, batch, 'fusion', True)
    assert all(torch.isfinite(value) for value in values.values())
    assert values['reliability'] > 0 and values['consistency'] > 0
    loss.backward()
    for parameter in (model.enhancers[0].response_order, model.enhancers[0].local_parameters.weight,
                      model.quality_heads[0][-2].weight, model.fusion_gata[0].router[-1].weight,
                      model.conditioners[0][0].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_restore_stage_skips_fusion():
    model = small_model()
    output = model(torch.rand(1, 1, 32, 32), torch.rand(1, 1, 32, 32), stage='restore', return_aux=True)
    assert output.fused is None and output.routes == ()
    output.restored_vi.mean().backward()
    assert all(parameter.grad is None for parameter in model.fusion_gata.parameters())


def test_ablation_forward_and_loss():
    model = small_model(use_response=False, use_reliability=False)
    batch = dict(vi=torch.rand(1, 1, 32, 32), ir=torch.rand(1, 1, 32, 32),
                 target_vi=torch.rand(1, 1, 32, 32), target_ir=torch.rand(1, 1, 32, 32))
    loss, components = training_loss(model, BRIFusionLoss(use_reliability=False), batch, 'fusion', False)
    loss.backward()
    assert components['reliability'] == components['ranking'] == components['consistency'] == 0


def test_separate_external_embeddings_and_input_validation():
    model = small_model()
    x = torch.rand(1, 1, 16, 16)
    model(x, x, (torch.rand(1, 512), torch.rand(1, 512)))
    with pytest.raises(ValueError, match='separate'):
        model(x, x, torch.rand(1, 512))
    with pytest.raises(ValueError, match='matching'):
        model(x, x[..., :10, :])


def test_ssim_identity():
    image = torch.rand(2, 1, 16, 16)
    assert abs(float(ssim_loss(image, image))) < 1e-6


def create_images(root, count=2):
    rng = np.random.default_rng(20)
    for folder in ('Vis', 'Inf', 'Vis_gt', 'Inf_gt'):
        (root / folder).mkdir(parents=True)
    for index in range(count):
        visible = rng.integers(20, 230, (33, 35, 3), dtype=np.uint8)
        infrared = rng.integers(20, 230, (33, 35), dtype=np.uint8)
        for folder, values in (('Vis_gt', visible), ('Inf_gt', infrared),
                               ('Vis', (visible * 0.6).astype(np.uint8)), ('Inf', infrared)):
            Image.fromarray(values).save(root / folder / f'{index}.png')


def test_dataset_alignment_and_shared_transforms(tmp_path):
    create_images(tmp_path)
    dataset = FusionDataset(tmp_path, 'synthetic', crop_size=32)
    sample = dataset[0]
    assert sample['vi'].shape == (1, 32, 32)
    torch.testing.assert_close(sample['vi'], sample['target_vi'])
    torch.testing.assert_close(sample['ir'], sample['target_ir'])
    Image.fromarray(np.zeros((16, 16), np.uint8)).save(tmp_path / 'Inf_gt' / '0.png')
    with pytest.raises(ValueError, match='dimensions'):
        FusionDataset(tmp_path, 'paired')[0]


def test_missing_pair_fails_early(tmp_path):
    create_images(tmp_path)
    (tmp_path / 'Inf' / '0.png').rename(tmp_path / 'Inf' / 'mismatch.png')
    with pytest.raises(ValueError, match='Unmatched'):
        FusionDataset(tmp_path)


def test_two_stage_checkpoint_resume_and_inference(tmp_path, monkeypatch):
    data = tmp_path / 'images'
    create_images(data)
    common = ['--data', str(data), '--epochs', '2', '--batch-size', '1', '--crop-size', '32',
              '--steps-per-epoch', '1', '--threads', '2', '--device', 'cpu']
    stage0 = train(common + ['--stage', 'restore', '--width', '4', '--output', str(tmp_path / 'restore')])
    epoch0_path = tmp_path / 'epoch0.pth'
    real_save = torch.save
    def save_and_keep_epoch0(state, path, *args, **kwargs):
        real_save(state, path, *args, **kwargs)
        if isinstance(state, dict) and state.get('stage') == 'fusion' and state['epoch'] == 0:
            real_save(state, epoch0_path)
    monkeypatch.setattr(torch, 'save', save_and_keep_epoch0)
    stage1 = train(common + ['--stage', 'fusion', '--init', str(stage0), '--output', str(tmp_path / 'fusion')])
    saved = load_checkpoint(stage1)
    assert saved['step'] == 2 and saved['stage'] == 'fusion'
    resumed = train(common + ['--stage', 'fusion', '--resume', str(epoch0_path), '--output', str(tmp_path / 'resumed')])
    resumed_state = load_checkpoint(resumed)
    for key, value in saved['model'].items():
        torch.testing.assert_close(value, resumed_state['model'][key], atol=0, rtol=0)
    results = tmp_path / 'results'
    report = infer(['--checkpoint', str(stage1), '--vi', str(data/'Vis'), '--ir', str(data/'Inf'),
                    '--output', str(results), '--device', 'cpu', '--warmup', '1', '--save-quality'])
    assert report['images'] == 2 and report['fps'] > 0
    assert Image.open(results/'0.png').size == (35, 33)
    assert (results/'quality/0_vi.png').exists()
    with pytest.raises(ValueError, match='stage-0'):
        infer(['--checkpoint', str(stage0), '--vi', str(data/'Vis'), '--ir', str(data/'Inf')])


def test_synthetic_training_with_all_ablations(tmp_path):
    create_images(tmp_path/'images', count=1)
    checkpoint = train(['--data', str(tmp_path/'images'), '--mode', 'synthetic', '--stage', 'fusion',
        '--from-scratch', '--epochs', '1', '--batch-size', '1', '--width', '4', '--crop-size', '32',
        '--no-response', '--no-reliability', '--no-intervention', '--threads', '2',
        '--device', 'cpu', '--output', str(tmp_path/'run')])
    state = load_checkpoint(checkpoint)
    assert not state['model_config']['use_response']
    assert not state['model_config']['use_reliability']
    assert not state['use_intervention']
