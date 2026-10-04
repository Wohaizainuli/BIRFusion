"""Response enhancement and reliability-conditioned sparse local fusion."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ChannelNorm(nn.Module):
    """Per-pixel channel normalization: no batch statistics, valid for 1x1 maps."""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        mean = x.float().mean(1, keepdim=True)
        var = x.float().var(1, keepdim=True, unbiased=False)
        return ((x.float() - mean) * torch.rsqrt(var + 1e-5)).to(x.dtype) * self.weight + self.bias


class ConvExpert(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1), nn.GELU(),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=dilation,
                      dilation=dilation, groups=channels * 2), nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1))

    def forward(self, x):
        return self.net(x)


def sparse_probabilities(logits, top_k):
    """Normalized Top-K weights and a differentiable Switch-style load loss."""
    probs = logits.float().softmax(-1)
    indices = probs.topk(top_k, dim=-1).indices
    selected = torch.zeros_like(probs).scatter_(-1, indices, 1)
    sparse = probs * selected
    sparse = sparse / sparse.sum(-1, keepdim=True).clamp_min(1e-8)
    usage = (selected / top_k).mean(0).detach()
    balance = probs.shape[-1] * (usage * probs.mean(0)).sum()
    return sparse.to(logits.dtype), balance


class ConditionalMoE(nn.Module):
    """DAMFusion-style shared experts with separate gates for VIS and IR."""
    def __init__(self, channels, condition_channels, experts=4, top_k=2):
        super().__init__()
        if not 1 <= top_k <= experts:
            raise ValueError('top_k must be between 1 and the number of experts')
        self.top_k = top_k
        self.norm = ChannelNorm(channels)
        self.gates = nn.ModuleList([nn.Linear(channels + condition_channels, experts) for _ in range(2)])
        self.experts = nn.ModuleList([ConvExpert(channels, 1 + i % 3) for i in range(experts)])

    def forward(self, x, condition, modality):
        descriptor = torch.cat([x.mean((2, 3)), condition.mean((2, 3))], 1)
        weights, balance = sparse_probabilities(self.gates[modality](descriptor), self.top_k)
        normalized = self.norm(x)
        residual = torch.zeros_like(x)
        for index, expert in enumerate(self.experts):
            selected = torch.where(weights[:, index] > 0)[0]
            if selected.numel():
                updates = expert(normalized[selected]) * weights[selected, index, None, None, None]
                residual = residual.index_add(0, selected, updates)
        return x + residual, balance


class BioInspiredAdaptiveResponseEnhancer(nn.Module):
    """Task adaptation of the response module supplied by the project author.

    Retains the Hill response, response mixing, multiscale spatial attention,
    and residual output. Adds local degradation conditioning and replaces the
    original batch-coupled normalization. See docs/METHOD.md for attribution.
    """
    def __init__(self, in_channels, feat_channels=16, num_responses=4,
                 condition_channels=16, initial_gain=0.05):
        super().__init__()
        self.num_responses = num_responses
        self.condition_channels = condition_channels
        orders = torch.linspace(1, min(num_responses, 8), num_responses)
        self.response_order = nn.Parameter(torch.logit((orders - 0.1) / 9.9))
        self.response_scale = nn.Parameter(torch.full((num_responses,), math.log(math.expm1(0.5))))
        self.local_parameters = nn.Conv2d(condition_channels, num_responses * 2, 1)
        self.response_gate = nn.Sequential(
            nn.Conv2d(in_channels + condition_channels, feat_channels, 3, padding=1),
            nn.SiLU(), nn.Conv2d(feat_channels, num_responses, 1))
        self.response_fusion = nn.Sequential(
            nn.Conv2d(in_channels * num_responses, feat_channels, 1),
            ChannelNorm(feat_channels), nn.PReLU())
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Conv2d(feat_channels, feat_channels * 2, 3, 2, 1),
                          ChannelNorm(feat_channels * 2), nn.PReLU()),
            nn.Sequential(nn.Conv2d(feat_channels * 2, feat_channels * 4, 3, 2, 1),
                          ChannelNorm(feat_channels * 4), nn.PReLU())])
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(feat_channels * 4, feat_channels * 4, 7, padding=3, groups=feat_channels * 4),
            nn.SiLU(), nn.Conv2d(feat_channels * 4, 1, 1), nn.Sigmoid())
        self.decoders = nn.ModuleList([
            nn.Sequential(nn.Conv2d(feat_channels * 6, feat_channels * 2, 3, padding=1), nn.PReLU()),
            nn.Sequential(nn.Conv2d(feat_channels * 3, feat_channels, 3, padding=1), nn.PReLU())])
        self.output_proj = nn.Conv2d(feat_channels, in_channels, 3, padding=1)
        self.residual_gain = nn.Parameter(torch.tensor(float(initial_gain)))

    @staticmethod
    def _normalize_response(response):
        # Each sample and channel is independent; eval does not depend on batch companions.
        mean = response.mean((-2, -1), keepdim=True)
        var = response.var((-2, -1), keepdim=True, unbiased=False)
        return (response - mean) * torch.rsqrt(var + 1e-5)

    def forward(self, x, condition=None):
        if condition is None:
            condition = x.new_zeros(x.shape[0], self.condition_channels, *x.shape[-2:])
        condition = F.interpolate(condition, size=x.shape[-2:], mode='bilinear', align_corners=False)
        weights = self.response_gate(torch.cat([x, condition], 1)).float().softmax(1)
        adjustment = self.local_parameters(condition).float().tanh()
        order_adjust, scale_adjust = adjustment.chunk(2, 1)
        positive = F.softplus(x.float()).clamp_min(1e-6)
        responses = []
        for index in range(self.num_responses):
            order = (0.1 + 9.9 * self.response_order[index].sigmoid()) * (1 + 0.25 * order_adjust[:, index:index+1])
            scale = (F.softplus(self.response_scale[index]) + 1e-4) * torch.exp(0.5 * scale_adjust[:, index:index+1])
            # Log-domain Hill response avoids overflow from x**order under AMP.
            response = torch.sigmoid(order * (positive.log() - scale.log()))
            responses.append(self._normalize_response(response) * weights[:, index:index+1])
        shallow = self.response_fusion(torch.cat(responses, 1).to(x.dtype))
        middle = self.encoders[0](shallow)
        deep = self.encoders[1](middle)
        deep = deep * self.spatial_attn(deep)
        y = self.decoders[0](torch.cat([F.interpolate(deep, size=middle.shape[-2:], mode='bilinear', align_corners=False), middle], 1))
        y = self.decoders[1](torch.cat([F.interpolate(y, size=shallow.shape[-2:], mode='bilinear', align_corners=False), shallow], 1))
        return x + self.residual_gain * self.output_proj(y)


def window_partition(x, window_size):
    b, c, h, w = x.shape
    ph, pw = (-h) % window_size, (-w) % window_size
    x = F.pad(x, (0, pw, 0, ph), mode='replicate')
    nh, nw = (h + ph) // window_size, (w + pw) // window_size
    tiles = x.reshape(b, c, nh, window_size, nw, window_size).permute(0, 2, 4, 1, 3, 5)
    return tiles.reshape(b * nh * nw, c, window_size, window_size), (b, c, h, w, nh, nw)


def window_merge(tiles, info):
    b, _, h, w, nh, nw = info
    c, size = tiles.shape[1], tiles.shape[-1]
    x = tiles.reshape(b, nh, nw, c, size, size).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, c, nh * size, nw * size)[..., :h, :w]


class ContextExpert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels * 2, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, 4, batch_first=True, dropout=0)
        self.ffn = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels))

    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normalized = self.norm(tokens)
        tokens = tokens + self.attention(normalized, normalized, normalized, need_weights=False)[0]
        tokens = tokens + self.ffn(self.norm(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class ReliabilityGuidedFusion(nn.Module):
    """Spatial reliability + content weighting, followed by true window Top-K dispatch."""
    def __init__(self, channels, top_k=2, window_size=4):
        super().__init__()
        if not 1 <= top_k <= 3 or window_size < 1:
            raise ValueError('Fusion needs 1 <= top_k <= 3 and a positive window size')
        self.top_k, self.window_size = top_k, window_size
        self.content = nn.Conv2d(channels * 2, 2, 1)
        self.router = nn.Sequential(nn.Conv2d(channels * 3 + 2, channels, 1), nn.SiLU(), nn.Conv2d(channels, 3, 1))
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels * 2, channels, 1), ConvExpert(channels)),
            ContextExpert(channels),
            nn.Conv2d(channels * 2, channels, 1)])
        self.residual_gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, vi, ir, q_vi, q_ir):
        quality = torch.cat([q_vi, q_ir], 1).clamp_min(1e-4)
        content = self.content(torch.cat([vi, ir], 1)).tanh()
        modality_weights = (content.float() + quality.float().log()).softmax(1).to(vi.dtype)
        base = modality_weights[:, :1] * vi + modality_weights[:, 1:] * ir
        local_logits = self.router(torch.cat([vi, ir, (vi - ir).abs(), quality], 1))
        logit_tiles, info = window_partition(local_logits, self.window_size)
        weights, balance = sparse_probabilities(logit_tiles.mean((2, 3)), self.top_k)
        tiles, _ = window_partition(torch.cat([modality_weights[:, :1] * vi, modality_weights[:, 1:] * ir], 1), self.window_size)
        updates = vi.new_zeros(tiles.shape[0], vi.shape[1], self.window_size, self.window_size)
        for index, expert in enumerate(self.experts):
            selected = torch.where(weights[:, index] > 0)[0]
            if selected.numel():
                value = expert(tiles[selected]) * weights[selected, index, None, None, None]
                updates = updates.index_add(0, selected, value)
        residual = window_merge(updates, info)
        b, _, _, _, nh, nw = info
        routes = weights.reshape(b, nh, nw, 3).permute(0, 3, 1, 2)
        return base + self.residual_gain * quality.mean(1, keepdim=True) * residual, balance, routes, modality_weights
