"""Reference-supervised reliability and localized intervention consistency."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .degradations import local_mean


def gradient(x):
    kernel = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    x = F.pad(x, (1, 1, 1, 1), mode='replicate')
    return F.conv2d(x, kernel).abs() + F.conv2d(x, kernel.transpose(2, 3)).abs()


def ssim_loss(x, y):
    x, y = x.float(), y.float()
    mx, my = local_mean(x, 11), local_mean(y, 11)
    vx = (local_mean(x.square(), 11) - mx.square()).clamp_min(0)
    vy = (local_mean(y.square(), 11) - my.square()).clamp_min(0)
    covariance = local_mean(x * y, 11) - mx * my
    numerator = (2 * mx * my + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)
    denominator = (mx.square() + my.square() + 0.01 ** 2) * (vx + vy + 0.03 ** 2)
    return 1 - (numerator / denominator).mean()


def masked_mean(values, mask):
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def reliability_target(observation, reference, tau=0.1):
    if tau <= 0:
        raise ValueError('Reliability temperature must be positive')
    discrepancy = local_mean((observation.float() - reference.float()).abs(), 7)
    return torch.exp(-discrepancy / tau).detach()


def reliability_ranking(before, after, target_before, target_after, mask, max_margin=0.1):
    # Ranking is valid only where the intervention actually worsened reference error.
    drop = (target_before - target_after).detach()
    eligible = mask * (drop > 1e-4).to(mask.dtype)
    violation = F.relu(after - before.detach() + drop.clamp(0, max_margin))
    return masked_mean(violation, eligible)


def intervention_consistency(before, after, mask, radius=12):
    affected = F.max_pool2d(mask, radius * 2 + 1, stride=1, padding=radius) if radius else mask
    return masked_mean((after - before.detach()).abs(), 1 - affected)


@dataclass
class LossConfig:
    pixel: float = 20.0
    gradient: float = 50.0
    ssim: float = 10.0
    restoration: float = 1.0
    reliability: float = 1.0
    ranking: float = 0.5
    consistency: float = 2.0
    balance: float = 0.01
    reliability_tau: float = 0.1
    exclusion_radius: int = 12


class BRIFusionLoss(nn.Module):
    def __init__(self, config=None, use_reliability=True):
        super().__init__()
        self.config = config or LossConfig()
        self.use_reliability = use_reliability

    def components(self, output, vi, ir, target_vi, target_ir, stage):
        zero = output.restored_vi.sum() * 0
        restoration = (F.l1_loss(output.restored_vi, target_vi) + F.l1_loss(output.restored_ir, target_ir)
                       + ssim_loss(output.restored_vi, target_vi) + ssim_loss(output.restored_ir, target_ir))
        result = dict(restoration=restoration, pixel=zero, gradient=zero, ssim=zero,
                      reliability=zero, balance=output.balance_loss)
        if stage == 'fusion':
            result['pixel'] = F.l1_loss(output.fused, torch.maximum(target_vi, target_ir))
            result['gradient'] = F.l1_loss(gradient(output.fused), torch.maximum(gradient(target_vi), gradient(target_ir)))
            result['ssim'] = ssim_loss(output.fused, target_vi) + ssim_loss(output.fused, target_ir)
        if self.use_reliability:
            qv = reliability_target(vi, target_vi, self.config.reliability_tau)
            qi = reliability_target(ir, target_ir, self.config.reliability_tau)
            result['reliability'] = torch.stack([F.mse_loss(q, t) for qs, t in
                ((output.reliability_vi, qv), (output.reliability_ir, qi)) for q in qs]).mean()
        return result

    def forward(self, output, batch, stage='fusion', intervened=None):
        vi, ir, tv, ti = (batch[k] for k in ('vi', 'ir', 'target_vi', 'target_ir'))
        components = self.components(output, vi, ir, tv, ti, stage)
        zero = output.restored_vi.sum() * 0
        components.update(ranking=zero, consistency=zero)
        if intervened is not None:
            other, other_vi, other_ir, mask_vi, mask_ir = intervened
            other_components = self.components(other, other_vi, other_ir, tv, ti, stage)
            for name, value in other_components.items():
                components[name] = 0.5 * (components[name] + value)
            if self.use_reliability:
                rankings = []
                for before_maps, after_maps, before_image, after_image, target, mask in (
                    (output.reliability_vi, other.reliability_vi, vi, other_vi, tv, mask_vi),
                    (output.reliability_ir, other.reliability_ir, ir, other_ir, ti, mask_ir)):
                    qb = reliability_target(before_image, target, self.config.reliability_tau)
                    qa = reliability_target(after_image, target, self.config.reliability_tau)
                    rankings.extend(reliability_ranking(b, a, qb, qa, mask) for b, a in zip(before_maps, after_maps))
                components['ranking'] = torch.stack(rankings).mean()
            if stage == 'fusion':
                components['consistency'] = intervention_consistency(output.fused, other.fused,
                    torch.maximum(mask_vi, mask_ir), self.config.exclusion_radius)
        total = sum(getattr(self.config, name) * value for name, value in components.items())
        return total, components
