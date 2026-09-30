from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DeformDescBranch(nn.Module):
    """Feature-side deformable descriptor residual used by E063-D0.

    The module predicts a bounded local sampling offset and a reliability value
    from normalized descriptors. It is independent as a component, but the
    validated E063-D0 method placed it before the unchanged OfficialSLiM
    matcher. This portable copy intentionally contains no SLiM dependency.
    """

    def __init__(
        self,
        max_channels: int = 768,
        rank: int = 16,
        max_scales: int = 6,
    ) -> None:
        super().__init__()
        self.max_channels = max_channels
        self.rank = rank
        self.down = nn.Parameter(torch.randn(max_scales, max_channels, rank) * 0.01)
        self.offset_head = nn.Parameter(torch.randn(max_scales, rank, 2) * 0.01)
        self.reliability_head = nn.Parameter(
            torch.randn(max_scales, rank, 1) * 0.01
        )
        self.mix = nn.Parameter(torch.zeros(max_scales))
        self.offset_scale = nn.Parameter(torch.zeros(max_scales))
        self.last_offsets: torch.Tensor | None = None
        self.last_reliability: torch.Tensor | None = None
        self.last_offset_abs = 0.0
        self.last_reliability_mean = 0.0

    def forward(self, x: torch.Tensor, scale_idx: int) -> torch.Tensor:
        channels = int(x.shape[1])
        if channels > self.max_channels:
            raise RuntimeError(
                f"channels {channels} exceed DeformDesc max_channels "
                f"{self.max_channels}"
            )
        if not 0 <= scale_idx < self.down.shape[0]:
            raise IndexError(f"scale_idx {scale_idx} is out of range")

        batch, _channels, height, width = x.shape
        feature = x.float()
        descriptor = F.normalize(feature, dim=1, eps=1e-6).permute(0, 2, 3, 1)
        latent = torch.einsum(
            "bhwc,cr->bhwr", descriptor, self.down[scale_idx, :channels]
        )
        offsets = torch.tanh(
            torch.einsum("bhwr,rd->bhwd", latent, self.offset_head[scale_idx])
        )
        reliability = torch.sigmoid(
            torch.einsum(
                "bhwr,ro->bhwo", latent, self.reliability_head[scale_idx]
            )
        )

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=x.device),
            torch.linspace(-1.0, 1.0, width, device=x.device),
            indexing="ij",
        )
        base_grid = torch.stack([xx, yy], dim=-1).view(
            1, height, width, 2
        ).expand(batch, -1, -1, -1)
        normalizer = x.new_tensor(
            [max(width - 1, 1), max(height - 1, 1)]
        ).view(1, 1, 1, 2)
        grid = base_grid + (
            2.0 * torch.tanh(self.offset_scale[scale_idx]) * offsets / normalizer
        )
        sampled = F.grid_sample(
            feature,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        self.last_offsets = offsets
        self.last_reliability = reliability
        self.last_offset_abs = float(offsets.detach().abs().mean())
        self.last_reliability_mean = float(reliability.detach().mean())
        residual = reliability.permute(0, 3, 1, 2) * (sampled - feature)
        return (
            feature + 0.10 * torch.tanh(self.mix[scale_idx]) * residual
        ).to(dtype=x.dtype)

