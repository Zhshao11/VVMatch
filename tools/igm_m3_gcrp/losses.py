from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F

from tools.prompt37_igm_hp1_common import warp_points


def centers(height: int, width: int, stride: int, device: torch.device) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    return torch.stack(((x + 0.5) * stride, (y + 0.5) * stride), -1).reshape(-1, 2)


def contiguous_mask(valid: torch.Tensor, ratio: float, generator: torch.Generator) -> torch.Tensor:
    batch, height, width = valid.shape
    result = torch.zeros_like(valid)
    target = int(math.ceil(valid[0].sum().item() * ratio))
    while int((result & valid).sum()) < target:
        y = int(torch.randint(max(height - 3, 1), (1,), generator=generator, device=valid.device))
        x = int(torch.randint(max(width - 3, 1), (1,), generator=generator, device=valid.device))
        result[:, y : y + 4, x : x + 4] = True
    return result & valid


def correspondence_indices(source_hw: tuple[int, int], target_hw: tuple[int, int], stride: int, homography: torch.Tensor, source_mask: torch.Tensor, target_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = centers(*source_hw, stride, homography.device)
    projected = warp_points(source[None], homography)[0]
    tx = torch.round(projected[:, 0] / stride - 0.5).long()
    ty = torch.round(projected[:, 1] / stride - 0.5).long()
    inside = source_mask.flatten()[...] & torch.isfinite(projected).all(-1)
    inside &= (tx >= 0) & (tx < target_hw[1]) & (ty >= 0) & (ty < target_hw[0])
    safe = ty.clamp(0, target_hw[0] - 1) * target_hw[1] + tx.clamp(0, target_hw[1] - 1)
    inside &= target_mask.flatten()[safe]
    return torch.where(inside)[0], safe[inside], projected[inside]


def variance_covariance(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = tokens.reshape(-1, tokens.shape[-1])
    std = torch.sqrt(flat.var(0, unbiased=False) + 1e-4)
    variance = F.relu(1.0 - std).mean()
    centered = flat - flat.mean(0)
    covariance = centered.T @ centered / max(flat.shape[0] - 1, 1)
    offdiag = covariance - torch.diag(torch.diag(covariance))
    return variance, offdiag.square().mean()


def compute_gcrp_loss(model: Any, output: dict[str, Any], homographies: torch.Tensor, valid0: dict[str, torch.Tensor], valid1: dict[str, torch.Tensor], mask_seed: int) -> tuple[torch.Tensor, dict[str, float]]:
    batch = homographies.shape[0]
    geometry = homographies
    mask_losses, corr_losses, geo_losses, vc_losses = [], [], [], []
    for sample_index in range(batch):
        generator = torch.Generator(device=homographies.device).manual_seed(mask_seed + sample_index)
        for scale_index, (name, stride) in enumerate((("fine", 4), ("coarse", 8), ("guide", 16))):
            s0, s1 = output[f"student_{name}0"][sample_index], output[f"student_{name}1"][sample_index]
            t0, t1 = output[f"target_{name}0"][sample_index], output[f"target_{name}1"][sample_index]
            hw0, hw1 = valid0[name].shape[-2:], valid1[name].shape[-2:]
            ids0, ids1, projected = correspondence_indices(hw0, hw1, stride, geometry[sample_index : sample_index + 1], valid0[name][sample_index], valid1[name][sample_index])
            if ids0.numel() == 0:
                continue
            if name == "coarse":
                mask0 = output["token_mask0"][sample_index].flatten()
                mask1 = output["token_mask1"][sample_index].flatten()
                lookup01 = {int(left): right for left, right in zip(ids0.tolist(), ids1.tolist())}
                masked_ids0 = ids0[mask0[ids0]]
                if masked_ids0.numel():
                    corresponding1 = torch.tensor([lookup01[int(index)] for index in masked_ids0], device=ids0.device)
                    predicted0 = F.normalize(model.predictor(s0[masked_ids0]), dim=-1)
                    mask_losses.append(1.0 - (predicted0 * t1[corresponding1]).sum(-1).mean())
                inverse = torch.linalg.inv(geometry[sample_index : sample_index + 1])
                rev0, rev1, _ = correspondence_indices(hw1, hw0, stride, inverse, valid1[name][sample_index], valid0[name][sample_index])
                lookup10 = {int(left): right for left, right in zip(rev0.tolist(), rev1.tolist())}
                masked_ids1 = rev0[mask1[rev0]]
                if masked_ids1.numel():
                    corresponding0 = torch.tensor([lookup10[int(index)] for index in masked_ids1], device=ids0.device)
                    predicted1 = F.normalize(model.predictor(s1[masked_ids1]), dim=-1)
                    mask_losses.append(1.0 - (predicted1 * t0[corresponding0]).sum(-1).mean())
                model.branch_counts["mask"] += int(masked_ids0.numel() + masked_ids1.numel()); model.branch_counts["predictor"] += 2
                target_xy = centers(*hw1, stride, s0.device)
                distance01 = torch.cdist(projected, target_xy)
                allowed01 = distance01 > 2 * stride
                allowed01[torch.arange(ids1.numel(), device=s0.device), ids1] = True
                similarity = (s0[ids0] @ t1.T / 0.07).masked_fill(~allowed01, -1e9)
                corr_losses.append(F.cross_entropy(similarity, ids1))
                source_xy = centers(*hw0, stride, s0.device)
                reverse_projected = warp_points(target_xy[None], inverse)[0][rev0]
                distance10 = torch.cdist(reverse_projected, source_xy)
                allowed10 = distance10 > 2 * stride
                allowed10[torch.arange(rev1.numel(), device=s0.device), rev1] = True
                reverse_similarity = (s1[rev0] @ t0.T / 0.07).masked_fill(~allowed10, -1e9)
                corr_losses.append(F.cross_entropy(reverse_similarity, rev1))
                model.branch_counts["contrast"] += 2
            if name == "guide":
                similarity = s0[ids0] @ t1.T / 0.07
                probability = torch.softmax(similarity, -1)
                target_xy = centers(*hw1, stride, similarity.device)
                expected = probability @ target_xy
                diagonal = math.sqrt((hw1[0] * stride) ** 2 + (hw1[1] * stride) ** 2)
                geo_losses.append(torch.linalg.vector_norm(expected - projected, dim=-1).mean() / diagonal)
                inverse = torch.linalg.inv(geometry[sample_index : sample_index + 1])
                rev0, rev1, reverse_projected = correspondence_indices(hw1, hw0, stride, inverse, valid1[name][sample_index], valid0[name][sample_index])
                reverse_probability = torch.softmax(s1[rev0] @ t0.T / 0.07, -1)
                expected_reverse = reverse_probability @ centers(*hw0, stride, similarity.device)
                diagonal_reverse = math.sqrt((hw0[0] * stride) ** 2 + (hw0[1] * stride) ** 2)
                geo_losses.append(torch.linalg.vector_norm(expected_reverse - reverse_projected, dim=-1).mean() / diagonal_reverse)
                model.branch_counts["geometry"] += 2
            variance, covariance = variance_covariance(torch.cat([s0, s1], 0))
            vc_losses.append(variance + covariance)
            model.branch_counts["variance_covariance"] += 1
    zero = sum(value.sum() * 0 for key, value in output.items() if isinstance(value, torch.Tensor))
    mask_loss = torch.stack(mask_losses).mean() if mask_losses else zero
    corr_loss = torch.stack(corr_losses).mean() if corr_losses else zero
    geo_loss = torch.stack(geo_losses).mean() if geo_losses else zero
    vc_loss = torch.stack(vc_losses).mean() if vc_losses else zero
    total = mask_loss + corr_loss + 0.2 * geo_loss + 0.05 * vc_loss
    return total, {"loss_total": float(total.detach()), "loss_mask": float(mask_loss.detach()), "loss_corr": float(corr_loss.detach()), "loss_geo": float(geo_loss.detach()), "loss_vc": float(vc_loss.detach())}
