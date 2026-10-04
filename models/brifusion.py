"""BRIFusion: four-scale restoration/fusion architecture adapted from DAMFusion."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .bri_layers import (BioInspiredAdaptiveResponseEnhancer, ConditionalMoE,
                         ConvExpert, ReliabilityGuidedFusion)


@dataclass
class FusionOutput:
    restored_vi: torch.Tensor
    restored_ir: torch.Tensor
    fused: torch.Tensor | None
    reliability_vi: tuple
    reliability_ir: tuple
    routes: tuple
    modality_weights: tuple
    balance_loss: torch.Tensor


class PyramidDecoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.refine = nn.ModuleList([ConvExpert(i * width) for i in (5, 4, 3, 2)])
        # Same pixel-shuffle scale progression as the original DAMFusion decoder.
        self.up = nn.ModuleList([
            nn.Sequential(nn.Conv2d(i * width, i * width, 3, padding=1), nn.PixelShuffle(2),
                          nn.Conv2d(i * width // 4, (i - 1) * width, 3, padding=1), nn.LeakyReLU(0.1))
            for i in (5, 4, 3, 2)])
        self.output = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, features):
        x = features[-1]
        for index, (refine, up) in enumerate(zip(self.refine, self.up)):
            if index:
                x = x + features[-index-1]
            x = up(x + refine(x))
        return self.output(x).sigmoid()


class BRIFusion(nn.Module):
    def __init__(self, width=16, condition_channels=16, num_responses=4,
                 encoder_experts=4, encoder_top_k=2, fusion_top_k=2,
                 window_size=4, use_response=True, use_reliability=True):
        super().__init__()
        if width < 4 or width % 4:
            raise ValueError('width must be a positive multiple of 4')
        self.config = dict(width=width, condition_channels=condition_channels, num_responses=num_responses,
                           encoder_experts=encoder_experts, encoder_top_k=encoder_top_k,
                           fusion_top_k=fusion_top_k, window_size=window_size,
                           use_response=use_response, use_reliability=use_reliability)
        self.use_response, self.use_reliability = use_response, use_reliability
        self.conditioners = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, condition_channels, 3, padding=1), nn.SiLU(),
                          nn.Conv2d(condition_channels, condition_channels, 3, padding=1)) for _ in range(2)])
        self.external_condition = nn.Linear(512, condition_channels, bias=False)
        self.stem = nn.Conv2d(1, width, 3, padding=1)
        self.encoder_moe = nn.ModuleList([ConditionalMoE(i * width, condition_channels, encoder_experts, encoder_top_k) for i in (1, 2, 3, 4)])
        # Retains the original Conv -> PixelUnshuffle -> Conv downsampling structure.
        self.down = nn.ModuleList([
            nn.Sequential(nn.Conv2d(i * width, i * width, 3, padding=1, bias=False), nn.PixelUnshuffle(2),
                          nn.Conv2d(i * width * 4, (i + 1) * width, 3, padding=1, bias=False), nn.LeakyReLU(0.1))
            for i in (1, 2, 3, 4)])
        self.enhancers = nn.ModuleList([
            BioInspiredAdaptiveResponseEnhancer(i * width, condition_channels=condition_channels,
                                                num_responses=num_responses) for i in (2, 3)]) if use_response else nn.ModuleList()
        self.quality_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(i * width + condition_channels, condition_channels, 3, padding=1), nn.SiLU(),
                          nn.Conv2d(condition_channels, 1, 1), nn.Sigmoid()) for i in (2, 3, 4, 5)]) if use_reliability else nn.ModuleList()
        self.fusion_gata = nn.ModuleList([ReliabilityGuidedFusion(i * width, fusion_top_k, window_size) for i in (2, 3, 4, 5)])
        self.decode_vi, self.decode_ir, self.decode_fi = (PyramidDecoder(width) for _ in range(3))

    def _encode(self, image, modality, external):
        condition = self.conditioners[modality](image)
        if external is not None:
            if external.shape != (image.shape[0], 512):
                raise ValueError('Each external modality embedding must have shape [B, 512]')
            condition = condition + self.external_condition(external.to(image))[:, :, None, None]
        x = self.stem(image)
        features, qualities, balances = [], [], []
        for index, (moe, down) in enumerate(zip(self.encoder_moe, self.down)):
            x, balance = moe(x, condition, modality)
            x = down(x)
            local = F.interpolate(condition, size=x.shape[-2:], mode='bilinear', align_corners=False)
            if self.use_response and index < 2:
                x = self.enhancers[index](x, local)
            q = self.quality_heads[index](torch.cat([x, local], 1)) if self.use_reliability else x.new_ones(x.shape[0], 1, *x.shape[-2:])
            features.append(x)
            qualities.append(q)
            balances.append(balance)
        return features, qualities, balances

    def forward(self, vi, ir, external_features=None, return_aux=False, stage='fusion'):
        if vi.ndim != 4 or vi.shape != ir.shape or vi.shape[1] != 1 or min(vi.shape[-2:]) < 1:
            raise ValueError('Aligned inputs must have matching [B, 1, H, W] shapes')
        if stage not in ('restore', 'fusion'):
            raise ValueError('stage must be restore or fusion')
        if external_features is None:
            external_features = (None, None)
        if not isinstance(external_features, (tuple, list)) or len(external_features) != 2:
            raise ValueError('Pass separate (VIS_embedding, IR_embedding), not their elementwise product')
        h, w = vi.shape[-2:]
        pad = (0, (-w) % 16, 0, (-h) % 16)
        vi, ir = (F.pad(x, pad, mode='replicate') for x in (vi, ir))
        fv, qv, balances = self._encode(vi, 0, external_features[0])
        fi, qi, other_balances = self._encode(ir, 1, external_features[1])
        balances.extend(other_balances)
        restored_vi, restored_ir = self.decode_vi(fv)[..., :h, :w], self.decode_ir(fi)[..., :h, :w]
        fused, routes, weights = None, [], []
        if stage == 'fusion':
            fused_features = []
            for block, v, i, vq, iq in zip(self.fusion_gata, fv, fi, qv, qi):
                feature, balance, route, modal_weight = block(v, i, vq, iq)
                fused_features.append(feature)
                balances.append(balance)
                routes.append(route)
                weights.append(modal_weight)
            fused = self.decode_fi(fused_features)[..., :h, :w]
        resize_quality = lambda maps: tuple(F.interpolate(q, size=vi.shape[-2:], mode='bilinear', align_corners=False)[..., :h, :w] for q in maps)
        result = FusionOutput(restored_vi, restored_ir, fused, resize_quality(qv), resize_quality(qi),
                              tuple(routes), tuple(weights), torch.stack(balances).mean())
        return result if return_aux else (restored_vi, restored_ir, fused, result.balance_loss)


Fusion = BRIFusion
