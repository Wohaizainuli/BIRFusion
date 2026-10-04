"""Training-only synthetic degradations and single-modality local interventions."""
import torch
from torch.nn import functional as F


def local_mean(x, kernel=7):
    return F.avg_pool2d(F.pad(x, (kernel // 2,) * 4, mode='replicate'), kernel, stride=1)


@torch.no_grad()
def compound_degrade(image, modality=0):
    output = image.clone()
    for index in range(image.shape[0]):
        x = image[index:index+1]
        # Preserve clean examples as part of the training distribution.
        if torch.rand(()).item() < 0.15:
            continue
        if modality == 0 and torch.rand(()).item() < 0.8:
            gamma = 1.2 + 1.8 * torch.rand((), device=x.device)
            gain = 0.4 + 0.6 * torch.rand((), device=x.device)
            x = x.clamp_min(1e-6).pow(gamma) * gain
        if torch.rand(()).item() < 0.5:
            contrast = 0.4 + 0.6 * torch.rand((), device=x.device)
            x = (x - x.mean()) * contrast + x.mean()
        if torch.rand(()).item() < 0.35:
            x = local_mean(x, 3)
        if modality == 1 and torch.rand(()).item() < 0.4:
            x = x + torch.randn_like(x[..., :1, :]) * 0.03
        sigma = 0.005 + 0.075 * torch.rand((), device=x.device)
        output[index:index+1] = (x + sigma * torch.randn_like(x)).clamp(0, 1)
    return output


@torch.no_grad()
def local_intervention(vi, ir, min_fraction=0.15, max_fraction=0.45):
    """Change only one modality and one rectangle per sample; return exact masks.

    Further degradation can occasionally reduce reference error. The criterion
    checks actual local reference discrepancy before applying ranking constraints.
    """
    if vi.shape != ir.shape or not 0 < min_fraction <= max_fraction < 1:
        raise ValueError('Invalid intervention shapes or region fractions')
    changed = [vi.clone(), ir.clone()]
    masks = [torch.zeros_like(vi[:, :1]), torch.zeros_like(ir[:, :1])]
    h, w = vi.shape[-2:]
    for index in range(vi.shape[0]):
        modality = int(torch.randint(2, ()).item())
        fraction = min_fraction + (max_fraction - min_fraction) * torch.rand(()).item()
        rh, rw = max(1, round(h * fraction)), max(1, round(w * fraction))
        top = int(torch.randint(h - rh + 1, ()).item())
        left = int(torch.randint(w - rw + 1, ()).item())
        original = changed[modality][index:index+1]
        operation = int(torch.randint(3, ()).item())
        if operation == 0:
            degraded = original + torch.randn_like(original) * (0.04 + 0.08 * torch.rand((), device=vi.device))
        elif operation == 1:
            degraded = original * (0.2 + 0.4 * torch.rand((), device=vi.device))
        else:
            degraded = local_mean(original, 5) + 0.025 * torch.randn_like(original)
        mask = masks[modality][index:index+1]
        mask[..., top:top+rh, left:left+rw] = 1
        changed[modality][index:index+1] = torch.where(mask.bool(), degraded.clamp(0, 1), original)
    return changed[0], changed[1], masks[0], masks[1]
