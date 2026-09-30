from __future__ import annotations

from typing import Any

import torch

from tools.prompt37_igm_hp1_common import PairSample, warp_points
from .model import IndependentGeometryMatcherM0R1


def _centers(height: int, width: int, stride: int, device: torch.device) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
    )
    return torch.stack(((x + 0.5) * stride, (y + 0.5) * stride), dim=-1).reshape(-1, 2)


def _directional_coarse_loss(
    categorical: torch.Tensor,
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
    homography: torch.Tensor,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
    target_valid_hw: tuple[int, int],
    stride: int = 8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    source_xy = _centers(*source_hw, stride, categorical.device)
    projected = warp_points(source_xy[None], homography)[0]
    valid_h, valid_w = target_valid_hw
    valid_projection = source_mask[0] & torch.isfinite(projected).all(dim=1)
    valid_projection &= (projected[:, 0] >= 0) & (projected[:, 0] < valid_w)
    valid_projection &= (projected[:, 1] >= 0) & (projected[:, 1] < valid_h)
    center_x = torch.round(projected[:, 0] / stride - 0.5).long()
    center_y = torch.round(projected[:, 1] / stride - 0.5).long()
    positive_mass = torch.zeros(categorical.shape[1], device=categorical.device)
    positive_count = torch.zeros(categorical.shape[1], dtype=torch.long, device=categorical.device)
    target_h, target_w = target_hw
    source_index = torch.arange(categorical.shape[1], device=categorical.device)
    for delta_y in (-1, 0, 1):
        for delta_x in (-1, 0, 1):
            x = center_x + delta_x
            y = center_y + delta_y
            inside = (x >= 0) & (x < target_w) & (y >= 0) & (y < target_h)
            safe_x = x.clamp(0, target_w - 1)
            safe_y = y.clamp(0, target_h - 1)
            target_index = safe_y * target_w + safe_x
            candidate_xy = torch.stack(((safe_x + 0.5) * stride, (safe_y + 0.5) * stride), dim=1)
            within = torch.linalg.vector_norm(candidate_xy - projected, dim=1) <= stride
            positive = valid_projection & inside & within & target_mask[0, target_index]
            positive_mass = positive_mass + categorical[0, source_index, target_index] * positive
            positive_count = positive_count + positive.long()
    matched = valid_projection & (positive_count > 0)
    unmatched = valid_projection & ~matched
    terms = []
    if matched.any():
        terms.append(-torch.log(positive_mass[matched].clamp_min(1e-8)).mean())
    if unmatched.any():
        terms.append(-torch.log(categorical[0, unmatched, -1].clamp_min(1e-8)).mean())
    loss = torch.stack(terms).sum() if terms else categorical.sum() * 0.0
    return loss, {
        "valid": int(valid_projection.sum()),
        "matched": int(matched.sum()),
        "unmatched": int(unmatched.sum()),
        "projected": projected,
        "positive_count": positive_count,
        "matched_mask": matched,
        "center_x": center_x,
        "center_y": center_y,
    }


def _charbonnier(delta: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(delta.square().sum(dim=-1) + epsilon**2)


def compute_igm_loss(
    model: IndependentGeometryMatcherM0R1,
    output: dict[str, Any],
    sample: PairSample,
) -> tuple[torch.Tensor, dict[str, float]]:
    source_mask = output["coarse_mask0"].flatten(1)
    target_mask = output["coarse_mask1"].flatten(1)
    source_hw = tuple(output["coarse0"].shape[-2:])
    target_hw = tuple(output["coarse1"].shape[-2:])
    forward, details = _directional_coarse_loss(
        output["source_assignment"], source_mask, target_mask,
        sample.homography, source_hw, target_hw, sample.valid_hw1,
    )
    reverse, _ = _directional_coarse_loss(
        output["target_assignment"],
        target_mask, source_mask, torch.linalg.inv(sample.homography),
        target_hw, source_hw, sample.valid_hw0,
    )
    coarse_loss = 0.5 * (forward + reverse)

    matched_indices = torch.where(details["matched_mask"])[0]
    if matched_indices.numel() == 0:
        fine_loss = output["fine0"].sum() * 0.0 + output["fine1"].sum() * 0.0
        cycle_loss = fine_loss
        median_transfer = float("inf")
    else:
        source_center = _centers(*source_hw, 8, output["source_assignment"].device)[matched_indices]
        target_x = details["center_x"][matched_indices].clamp(0, target_hw[1] - 1)
        target_y = details["center_y"][matched_indices].clamp(0, target_hw[0] - 1)
        target_center = torch.stack(((target_x + 0.5) * 8, (target_y + 0.5) * 8), dim=1)
        batch_index = torch.zeros_like(matched_indices)
        refined0, refined1, _, _ = model.refine(
            output["fine0"], output["fine1"], batch_index, source_center, target_center,
            sample.valid_hw0, sample.valid_hw1,
        )
        gt_target = warp_points(source_center[None], sample.homography)[0]
        gt_source = warp_points(target_center[None], torch.linalg.inv(sample.homography))[0]
        fine_loss = 0.5 * (_charbonnier(refined0 - gt_source).mean() + _charbonnier(refined1 - gt_target).mean()) / 4.0
        warped0 = warp_points(refined0[None], sample.homography)[0]
        warped1 = warp_points(refined1[None], torch.linalg.inv(sample.homography))[0]
        cycle_error = 0.5 * (_charbonnier(warped0 - refined1) + _charbonnier(warped1 - refined0))
        cycle_loss = cycle_error.mean() / 4.0
        median_transfer = float(cycle_error.detach().median())
    total = coarse_loss + fine_loss + 0.1 * cycle_loss
    metrics = {
        "loss_total": float(total.detach()),
        "loss_coarse": float(coarse_loss.detach()),
        "loss_fine": float(fine_loss.detach()),
        "loss_cycle": float(cycle_loss.detach()),
        "median_refined_transfer_px": median_transfer,
        "coarse_valid": float(details["valid"]),
        "coarse_matched": float(details["matched"]),
        "source_assignment_entropy": float(-(output["source_assignment"].clamp_min(1e-8) * output["source_assignment"].clamp_min(1e-8).log()).sum(-1).mean().detach()),
        "target_assignment_entropy": float(-(output["target_assignment"].clamp_min(1e-8) * output["target_assignment"].clamp_min(1e-8).log()).sum(-1).mean().detach()),
        "source_null_argmax_rate": float((output["source_assignment"].argmax(-1) == output["source_assignment"].shape[-1] - 1).float().mean().detach()),
        "target_null_argmax_rate": float((output["target_assignment"].argmax(-1) == output["target_assignment"].shape[-1] - 1).float().mean().detach()),
    }
    return total, metrics
