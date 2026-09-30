from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = 8 if channels % 8 == 0 else 4
    return nn.GroupNorm(groups, channels)


class SeparableResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        hidden = out_channels * 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels
        )
        self.expand = nn.Conv2d(in_channels, hidden, 1)
        self.project = nn.Conv2d(hidden, out_channels, 1)
        self.norm = _group_norm(out_channels)
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.depthwise(x)
        x = F.gelu(self.expand(x))
        x = self.norm(self.project(x))
        return F.gelu(x + residual)


class IndependentPyramid(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), _group_norm(32), nn.GELU()
        )
        self.stage2 = nn.Sequential(
            SeparableResidualBlock(32, 64, stride=2),
            SeparableResidualBlock(64, 64),
        )
        self.stage3 = nn.Sequential(
            SeparableResidualBlock(64, 128, stride=2),
            SeparableResidualBlock(128, 128),
        )
        self.stage4 = nn.Sequential(
            SeparableResidualBlock(128, 192, stride=2),
            SeparableResidualBlock(192, 192),
        )
        self.lateral4 = nn.Conv2d(192, 128, 1)
        self.lateral3 = nn.Conv2d(128, 128, 1)
        self.coarse_out = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), _group_norm(128))
        self.coarse_to_fine = nn.Conv2d(128, 64, 1)
        self.lateral2 = nn.Conv2d(64, 64, 1)
        self.fine_out = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), _group_norm(64))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x2 = self.stage2(self.stem(image))
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        coarse = self.lateral3(x3) + F.interpolate(
            self.lateral4(x4), size=x3.shape[-2:], mode="bilinear", align_corners=False
        )
        coarse = self.coarse_out(coarse)
        fine = self.lateral2(x2) + F.interpolate(
            self.coarse_to_fine(coarse), size=x2.shape[-2:], mode="bilinear", align_corners=False
        )
        return coarse, self.fine_out(fine)


class LinearAttention(nn.Module):
    def __init__(self, width: int = 128, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def _split(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, _ = value.shape
        return value.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_mask: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        q = F.elu(self._split(self.q(query))) + 1.0
        k = F.elu(self._split(self.k(context))) + 1.0
        v = self._split(self.v(context))
        cmask = context_mask[:, None, :, None].to(q.dtype)
        qmask = query_mask[:, None, :, None].to(q.dtype)
        k = k * cmask
        v = v * cmask
        kv = torch.einsum("bhld,bhle->bhde", k, v)
        normalizer = torch.einsum("bhld,bhd->bhl", q, k.sum(dim=2)).clamp_min(1e-6)
        output = torch.einsum("bhld,bhde->bhle", q, kv) / normalizer[..., None]
        output = output * qmask
        output = output.transpose(1, 2).reshape(query.shape)
        return self.out(output)


class PairInteractionBlock(nn.Module):
    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.position = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.self_norm = nn.LayerNorm(width)
        self.cross_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.self_attention = LinearAttention(width, 4)
        self.cross_attention = LinearAttention(width, 4)
        self.ffn = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def positional_mix(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        feature = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)
        return tokens + self.position(feature).flatten(2).transpose(1, 2)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
        source_hw: tuple[int, int],
        target_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.positional_mix(source, *source_hw)
        target = self.positional_mix(target, *target_hw)
        source = source + self.self_attention(
            self.self_norm(source), self.self_norm(source), source_mask, source_mask
        )
        target = target + self.self_attention(
            self.self_norm(target), self.self_norm(target), target_mask, target_mask
        )
        source_before, target_before = source, target
        source = source + self.cross_attention(
            self.cross_norm(source_before), self.cross_norm(target_before), source_mask, target_mask
        )
        target = target + self.cross_attention(
            self.cross_norm(target_before), self.cross_norm(source_before), target_mask, source_mask
        )
        source = source + self.ffn(self.ffn_norm(source))
        target = target + self.ffn(self.ffn_norm(target))
        return source * source_mask[..., None], target * target_mask[..., None]


@dataclass
class MatchOutput:
    source_xy: torch.Tensor
    target_xy: torch.Tensor
    confidence: torch.Tensor
    source_index: torch.Tensor
    target_index: torch.Tensor


class IndependentGeometryMatcherM0R1(nn.Module):
    coarse_stride = 8
    fine_stride = 4
    patch_size = 9

    def __init__(self) -> None:
        super().__init__()
        self.encoder = IndependentPyramid()
        self.interaction = nn.ModuleList([PairInteractionBlock(), PairInteractionBlock()])
        self.source_projection = nn.Linear(128, 128, bias=False)
        self.target_projection = nn.Linear(128, 128, bias=False)
        self.source_matchability = nn.Linear(128, 1)
        self.target_matchability = nn.Linear(128, 1)
        self.fine_projection = nn.Conv2d(64, 64, 1)
        self.register_buffer("temperature", torch.tensor(0.1), persistent=True)
        self.branch_counts: dict[str, int] = {
            "encoder": 0,
            "interaction": 0,
            "assignment": 0,
            "refinement": 0,
        }

    @staticmethod
    def _token_centers(height: int, width: int, stride: int, device: torch.device) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
        )
        return torch.stack(((x + 0.5) * stride, (y + 0.5) * stride), dim=-1).reshape(-1, 2)

    def encode_pair(self, image0: torch.Tensor, image1: torch.Tensor) -> dict[str, torch.Tensor]:
        coarse0, fine0 = self.encoder(image0)
        coarse1, fine1 = self.encoder(image1)
        self.branch_counts["encoder"] += 1
        return {"coarse0": coarse0, "coarse1": coarse1, "fine0": fine0, "fine1": fine1}

    def interact(
        self,
        coarse0: torch.Tensor,
        coarse1: torch.Tensor,
        mask0: torch.Tensor,
        mask1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h0, w0 = coarse0.shape[-2:]
        h1, w1 = coarse1.shape[-2:]
        source = coarse0.flatten(2).transpose(1, 2)
        target = coarse1.flatten(2).transpose(1, 2)
        source_mask = mask0.flatten(1).bool()
        target_mask = mask1.flatten(1).bool()
        for block in self.interaction:
            source, target = block(
                source, target, source_mask, target_mask, (h0, w0), (h1, w1)
            )
        self.branch_counts["interaction"] += len(self.interaction)
        return source, target

    def assign(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        source_descriptor = F.normalize(self.source_projection(source), dim=-1)
        target_descriptor = F.normalize(self.target_projection(target), dim=-1)
        logits = torch.einsum("bic,bjc->bij", source_descriptor, target_descriptor) / self.temperature
        return self.categorical_assignment(
            logits,
            self.source_matchability(source).float(),
            self.target_matchability(target).float(),
            source_mask,
            target_mask,
        )

    def categorical_assignment(
        self,
        logits: torch.Tensor,
        source_null_logits: torch.Tensor,
        target_null_logits: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        valid = source_mask[:, :, None] & target_mask[:, None, :]
        masked_logits = logits.float().masked_fill(~valid, -torch.inf)
        source_categorical_logits = torch.cat((masked_logits, source_null_logits), dim=2)
        target_categorical_logits = torch.cat((masked_logits.transpose(1, 2), target_null_logits), dim=2)
        source_assignment = F.softmax(source_categorical_logits, dim=2)
        target_assignment = F.softmax(target_categorical_logits, dim=2)
        source_assignment = torch.where(
            source_mask[:, :, None], source_assignment,
            F.one_hot(
                torch.full_like(source_mask, logits.shape[2], dtype=torch.long),
                num_classes=logits.shape[2] + 1,
            ).to(source_assignment.dtype),
        )
        target_assignment = torch.where(
            target_mask[:, :, None], target_assignment,
            F.one_hot(
                torch.full_like(target_mask, logits.shape[1], dtype=torch.long),
                num_classes=logits.shape[1] + 1,
            ).to(target_assignment.dtype),
        )
        source_pair = source_assignment[:, :, :-1]
        target_pair = target_assignment[:, :, :-1]
        pair = torch.sqrt((source_pair * target_pair.transpose(1, 2)).clamp_min(0.0))
        source_null = source_assignment[:, :, -1:]
        target_null = target_assignment[:, :, -1:]
        self.branch_counts["assignment"] += 1
        return {
            "logits": masked_logits,
            "pair": pair,
            "source_null": source_null,
            "target_null": target_null.transpose(1, 2),
            "assignment": source_assignment,
            "source_assignment": source_assignment,
            "target_assignment": target_assignment,
        }

    @staticmethod
    def mutual_matches(
        source_assignment: torch.Tensor,
        target_assignment: torch.Tensor,
        maximum: int = 2048,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if source_assignment.shape[0] != 1 or target_assignment.shape[0] != 1:
            raise ValueError("inference currently requires batch size one")
        source_count = source_assignment.shape[1]
        target_count = target_assignment.shape[1]
        source_choice = source_assignment[0].argmax(dim=1)
        target_choice = target_assignment[0].argmax(dim=1)
        source_index = torch.arange(source_count, device=source_assignment.device)
        non_null = source_choice < target_count
        safe_target = source_choice.clamp_max(target_count - 1)
        mutual = non_null & (target_choice[safe_target] == source_index)
        mutual &= target_choice[safe_target] < source_count
        source_index = source_index[mutual]
        target_index = source_choice[mutual]
        confidence = torch.sqrt(
            source_assignment[0, source_index, target_index]
            * target_assignment[0, target_index, source_index]
        )
        if confidence.numel() > maximum:
            order = torch.argsort(confidence, descending=True, stable=True)[:maximum]
            source_index, target_index, confidence = (
                source_index[order], target_index[order], confidence[order]
            )
        return (
            torch.zeros_like(source_index), source_index, target_index, confidence
        )

    @staticmethod
    def _patch_grid(
        centers_xy: torch.Tensor,
        height: int,
        width: int,
        radius: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        offset_y, offset_x = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=centers_xy.device),
            torch.arange(-radius, radius + 1, device=centers_xy.device),
            indexing="ij",
        )
        offsets = torch.stack((offset_x, offset_y), dim=-1).to(centers_xy.dtype)
        absolute = centers_xy[:, None, None, :] + offsets[None]
        normalized = absolute.clone()
        normalized[..., 0] = 2.0 * normalized[..., 0] / max(width - 1, 1) - 1.0
        normalized[..., 1] = 2.0 * normalized[..., 1] / max(height - 1, 1) - 1.0
        return normalized, absolute

    def refine(
        self,
        fine0: torch.Tensor,
        fine1: torch.Tensor,
        batch_index: torch.Tensor,
        source_xy: torch.Tensor,
        target_xy: torch.Tensor,
        valid_hw0: tuple[int, int] | None = None,
        valid_hw1: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if source_xy.numel() == 0:
            empty = source_xy.new_zeros((0,))
            return source_xy, target_xy, empty, empty
        fine0 = self.fine_projection(fine0)
        fine1 = self.fine_projection(fine1)
        source_f = source_xy / self.fine_stride - 0.5
        target_f = target_xy / self.fine_stride - 0.5
        grid0, absolute0 = self._patch_grid(source_f, fine0.shape[-2], fine0.shape[-1])
        grid1, absolute1 = self._patch_grid(target_f, fine1.shape[-2], fine1.shape[-1])
        patch0 = self._sample_patches(fine0, batch_index, grid0)
        patch1 = self._sample_patches(fine1, batch_index, grid1)
        patch0 = F.normalize(patch0.flatten(2).transpose(1, 2), dim=-1)
        patch1 = F.normalize(patch1.flatten(2).transpose(1, 2), dim=-1)
        correlation = torch.einsum("nic,njc->nij", patch0, patch1) / 0.1
        if valid_hw0 is not None and valid_hw1 is not None:
            valid0 = ((absolute0[..., 0] + 0.5) * self.fine_stride < valid_hw0[1])
            valid0 &= ((absolute0[..., 1] + 0.5) * self.fine_stride < valid_hw0[0])
            valid0 &= (absolute0 >= 0).all(dim=-1)
            valid1 = ((absolute1[..., 0] + 0.5) * self.fine_stride < valid_hw1[1])
            valid1 &= ((absolute1[..., 1] + 0.5) * self.fine_stride < valid_hw1[0])
            valid1 &= (absolute1 >= 0).all(dim=-1)
            valid_pair = valid0.flatten(1)[:, :, None] & valid1.flatten(1)[:, None, :]
            correlation = correlation.masked_fill(~valid_pair, -1e9)
        else:
            valid_pair = torch.ones_like(correlation, dtype=torch.bool)
        dual = F.softmax(correlation, dim=2) * F.softmax(correlation, dim=1)
        dual = dual * valid_pair
        joint = dual / dual.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        source_probability = joint.sum(dim=2)
        target_probability = joint.sum(dim=1)
        absolute0 = absolute0.reshape(source_xy.shape[0], -1, 2)
        absolute1 = absolute1.reshape(target_xy.shape[0], -1, 2)
        refined0 = torch.sum(source_probability[..., None] * absolute0, dim=1)
        refined1 = torch.sum(target_probability[..., None] * absolute1, dim=1)
        refined0 = (refined0 + 0.5) * self.fine_stride
        refined1 = (refined1 + 0.5) * self.fine_stride
        self.branch_counts["refinement"] += 1
        return refined0, refined1, source_probability.max(dim=1).values, target_probability.max(dim=1).values

    @staticmethod
    def _sample_patches(
        feature: torch.Tensor, batch_index: torch.Tensor, grid: torch.Tensor
    ) -> torch.Tensor:
        patches = torch.empty(
            grid.shape[0], feature.shape[1], grid.shape[1], grid.shape[2],
            device=feature.device, dtype=feature.dtype,
        )
        for batch in batch_index.unique(sorted=True):
            selected = torch.where(batch_index == batch)[0]
            count = selected.numel()
            merged_grid = grid[selected].reshape(1, count * grid.shape[1], grid.shape[2], 2)
            sampled = F.grid_sample(
                feature[batch : batch + 1], merged_grid, mode="bilinear",
                padding_mode="zeros", align_corners=True,
            )
            patches[selected] = sampled.reshape(
                feature.shape[1], count, grid.shape[1], grid.shape[2]
            ).permute(1, 0, 2, 3)
        return patches

    @staticmethod
    def coarse_center_mask(pixel_mask: torch.Tensor, height: int, width: int, stride: int = 8) -> torch.Tensor:
        valid_height = pixel_mask.any(dim=2).sum(dim=1)
        valid_width = pixel_mask.any(dim=1).sum(dim=1)
        center_y = (torch.arange(height, device=pixel_mask.device) + 0.5) * stride
        center_x = (torch.arange(width, device=pixel_mask.device) + 0.5) * stride
        return (center_y[None, :, None] < valid_height[:, None, None]) & (
            center_x[None, None, :] < valid_width[:, None, None]
        )

    @staticmethod
    def stable_filter_and_deduplicate(
        source_xy: torch.Tensor,
        target_xy: torch.Tensor,
        confidence: torch.Tensor,
        source_index: torch.Tensor,
        target_index: torch.Tensor,
        valid_hw0: tuple[int, int],
        valid_hw1: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = torch.isfinite(source_xy).all(dim=1) & torch.isfinite(target_xy).all(dim=1)
        valid &= (source_xy >= 0).all(dim=1) & (target_xy >= 0).all(dim=1)
        valid &= (source_xy[:, 0] < valid_hw0[1]) & (source_xy[:, 1] < valid_hw0[0])
        valid &= (target_xy[:, 0] < valid_hw1[1]) & (target_xy[:, 1] < valid_hw1[0])
        candidates = torch.where(valid)[0]
        order = candidates[torch.argsort(confidence[candidates], descending=True, stable=True)]
        keep: list[int] = []
        seen_source: set[tuple[float, float]] = set()
        seen_target: set[tuple[float, float]] = set()
        for item in order.tolist():
            source_key = tuple(float(value) for value in source_xy[item].tolist())
            target_key = tuple(float(value) for value in target_xy[item].tolist())
            if source_key in seen_source or target_key in seen_target:
                continue
            seen_source.add(source_key)
            seen_target.add(target_key)
            keep.append(item)
        selected = torch.tensor(keep, dtype=torch.long, device=source_xy.device)
        return source_xy[selected], target_xy[selected], confidence[selected], source_index[selected], target_index[selected]

    def forward(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor,
        mask0: torch.Tensor,
        mask1: torch.Tensor,
        return_matches: bool = True,
    ) -> dict[str, Any]:
        features = self.encode_pair(image0, image1)
        coarse_mask0 = self.coarse_center_mask(mask0, *features["coarse0"].shape[-2:])
        coarse_mask1 = self.coarse_center_mask(mask1, *features["coarse1"].shape[-2:])
        source, target = self.interact(
            features["coarse0"], features["coarse1"], coarse_mask0, coarse_mask1
        )
        assignment = self.assign(source, target, coarse_mask0.flatten(1), coarse_mask1.flatten(1))
        output: dict[str, Any] = {
            **features,
            "source_state": source,
            "target_state": target,
            "coarse_mask0": coarse_mask0,
            "coarse_mask1": coarse_mask1,
            **assignment,
        }
        if not return_matches:
            return output
        batch_index, source_index, target_index, confidence = self.mutual_matches(
            assignment["source_assignment"], assignment["target_assignment"]
        )
        h0, w0 = features["coarse0"].shape[-2:]
        h1, w1 = features["coarse1"].shape[-2:]
        centers0 = self._token_centers(h0, w0, self.coarse_stride, image0.device)[source_index]
        centers1 = self._token_centers(h1, w1, self.coarse_stride, image1.device)[target_index]
        refined0, refined1, peak0, peak1 = self.refine(
            features["fine0"], features["fine1"], batch_index, centers0, centers1,
            (int(mask0.any(dim=2).sum(dim=1)[0]), int(mask0.any(dim=1).sum(dim=1)[0])),
            (int(mask1.any(dim=2).sum(dim=1)[0]), int(mask1.any(dim=1).sum(dim=1)[0])),
        )
        final_confidence = confidence * peak0 * peak1
        valid_hw0 = (int(mask0.any(dim=2).sum(dim=1)[0]), int(mask0.any(dim=1).sum(dim=1)[0]))
        valid_hw1 = (int(mask1.any(dim=2).sum(dim=1)[0]), int(mask1.any(dim=1).sum(dim=1)[0]))
        refined0, refined1, final_confidence, source_index, target_index = self.stable_filter_and_deduplicate(
            refined0, refined1, final_confidence, source_index, target_index, valid_hw0, valid_hw1
        )
        output["matches"] = MatchOutput(
            source_xy=refined0,
            target_xy=refined1,
            confidence=final_confidence,
            source_index=source_index,
            target_index=target_index,
        )
        return output
