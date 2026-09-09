# SPDX-License-Identifier: Apache-2.0
"""Vision tower and aligner for DeepSeek-V4-Flash-Vision-Exp.

Faithful port of the reference implementation shipped in the checkpoint at
``inference/vision.py``. Deliberately built from plain ``torch.nn`` modules and
``F.scaled_dot_product_attention``: the tower and aligner are BF16 in the
checkpoint (259 + 4 tensors, no scale tensors), and DeepseekV4FP8Config only
attaches quant methods to ``LinearBase``/``RoutedExperts``, so plain modules can
never be quantized by accident. This matches the deepseek_ocr2 precedent.

The tower runs one image per call with full bidirectional attention and 2D RoPE;
there is no CLS token, no learned position embedding, and no windowing.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn


@lru_cache(8)
def _cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    """2D RoPE tables: the first half of each head's rotary block is driven by
    the patch row, the second half by the column."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class VisionRMSNorm(nn.Module):
    # eps is 1e-6 here, NOT the LM's rms_norm_eps (1e-20): the reference Block
    # constructs RMSNorm(dim) with no eps argument, taking this default.
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Linear(3 * patch_size**2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-major flatten: index = c*patch^2 + py*patch + px.
        return self.proj(x.flatten(1))


class VisionAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wqkv = nn.Linear(dim, 3 * dim)
        self.wo = nn.Linear(dim, dim)

    def forward(self, x, cos, sin):
        n = x.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim)
                   for t in self.wqkv(x).chunk(3, dim=-1))
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        return self.wo(o.transpose(0, 1).reshape(n, -1))


class VisionMLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, 2 * inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)

    def forward(self, x):
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class VisionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, inter_dim: int):
        super().__init__()
        self.norm1 = VisionRMSNorm(dim)
        self.attn = VisionAttention(dim, n_heads)
        self.norm2 = VisionRMSNorm(dim)
        self.mlp = VisionMLP(dim, inter_dim)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class DeepseekV4ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.vision_dim
        n_heads = config.vision_n_heads
        self.rope_dim = dim // n_heads // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = PatchEmbed(dim, config.vision_patch_size)
        self.blocks = nn.ModuleList([
            VisionBlock(dim, n_heads, config.vision_inter_dim)
            for _ in range(config.vision_n_layers)
        ])
        self.norm = VisionRMSNorm(dim)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = _cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        # _cos_sin builds on CPU and is lru_cached; move to the activation device.
        cos = cos.to(x.device)
        sin = sin.to(x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class DeepseekV4Aligner(nn.Module):
    """3x3 space-to-depth (pixel-unshuffle) then a 2-layer GELU MLP to LM width."""

    def __init__(self, config):
        super().__init__()
        self.r = config.vision_downsample_ratio
        self.w1 = nn.Linear(config.vision_dim * self.r**2, config.hidden_size)
        self.w2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.r
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        # Zero-pad right/bottom to a multiple of r, matching the reference.
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))
