from __future__ import annotations

import math
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
        return coarse, self.fine_out(fine), x4


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


class AdaptiveSpanInteractionBlock(nn.Module):
    def __init__(self, width: int = 128, heads: int = 4) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.position = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.self_norm = nn.LayerNorm(width)
        self.cross_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.self_attention = LinearAttention(width, heads)
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.offset_bias = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, heads))
        self.ffn = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def _position(self, tokens: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        feature = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], *hw)
        return tokens + self.position(feature).flatten(2).transpose(1, 2)

    def _sample(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        context_hw: tuple[int, int],
        mu: torch.Tensor,
        radius: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, channels = context.shape
        h, w = context_hw
        feature = context.transpose(1, 2).reshape(batch, channels, h, w)
        mask_feature = context_mask.reshape(batch, 1, h, w).float()
        axis = torch.linspace(-1.0, 1.0, 6, device=context.device, dtype=context.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        normalized_offset = torch.stack((xx, yy), dim=-1).reshape(1, 1, 36, 2)
        absolute = mu[:, :, None, :] + normalized_offset * radius[:, :, None, :]
        grid = absolute.clone()
        grid[..., 0] = 2.0 * grid[..., 0] / max(w - 1, 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / max(h - 1, 1) - 1.0
        merged = grid.reshape(batch, length * 6, 6, 2)
        sampled = F.grid_sample(feature, merged, mode="bilinear", padding_mode="zeros", align_corners=True)
        sampled = sampled.reshape(batch, channels, length, 6, 6).permute(0, 2, 3, 4, 1).reshape(batch, length, 36, channels)
        sampled_mask = F.grid_sample(mask_feature, merged, mode="nearest", padding_mode="zeros", align_corners=True)
        sampled_mask = sampled_mask.reshape(batch, 1, length, 6, 6).permute(0, 2, 3, 4, 1).reshape(batch, length, 36).bool()
        return sampled, sampled_mask, normalized_offset.expand(batch, length, -1, -1)

    def self_update(
        self,
        query: torch.Tensor,
        query_mask: torch.Tensor,
        query_hw: tuple[int, int],
    ) -> torch.Tensor:
        query = self._position(query, query_hw)
        query = query + self.self_attention(
            self.self_norm(query), self.self_norm(query), query_mask, query_mask
        )
        return query * query_mask[..., None]

    def cross_update(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_mask: torch.Tensor,
        context_mask: torch.Tensor,
        context_hw: tuple[int, int],
        mu: torch.Tensor,
        radius: torch.Tensor,
        covisibility: torch.Tensor,
        cross_multiplier: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        normalized_query = self.cross_norm(query)
        sampled, sampled_mask, relative = self._sample(context, context_mask, context_hw, mu, radius)
        batch, length = query.shape[:2]
        q = self.q(normalized_query).reshape(batch, length, self.heads, self.head_dim)
        k = self.k(sampled).reshape(batch, length, 36, self.heads, self.head_dim)
        v = self.v(sampled).reshape(batch, length, 36, self.heads, self.head_dim)
        logits = torch.einsum("blhd,blshd->blhs", q, k) / math.sqrt(self.head_dim)
        bias = self.offset_bias(relative).permute(0, 1, 3, 2)
        logits = logits + bias
        valid = sampled_mask[:, :, None, :]
        logits = logits.masked_fill(~valid, -1e9)
        attention = F.softmax(logits, dim=-1) * valid
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        message = torch.einsum("blhs,blshd->blhd", attention, v).reshape(batch, length, self.width)
        message = self.out(message) * covisibility * float(cross_multiplier)
        query = query + message
        query = query + self.ffn(self.ffn_norm(query))
        query = query * query_mask[..., None]
        return query, {"message": message, "sample_mask": sampled_mask, "attention": attention}

    def forward_direction(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_mask: torch.Tensor,
        context_mask: torch.Tensor,
        query_hw: tuple[int, int],
        context_hw: tuple[int, int],
        mu: torch.Tensor,
        radius: torch.Tensor,
        covisibility: torch.Tensor,
        cross_multiplier: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        query = self.self_update(query, query_mask, query_hw)
        return self.cross_update(
            query, context, query_mask, context_mask, context_hw,
            mu, radius, covisibility, cross_multiplier,
        )


class IndependentGeometryMatcherM1VASI(nn.Module):
    coarse_stride = 8
    fine_stride = 4
    patch_size = 9

    def __init__(self) -> None:
        super().__init__()
        self.encoder = IndependentPyramid()
        after_encoder_rng = torch.get_rng_state()
        _parent_interaction_rng = nn.ModuleList([PairInteractionBlock(), PairInteractionBlock()])
        self.source_projection = nn.Linear(128, 128, bias=False)
        self.target_projection = nn.Linear(128, 128, bias=False)
        self.source_matchability = nn.Linear(128, 1)
        self.target_matchability = nn.Linear(128, 1)
        self.fine_projection = nn.Conv2d(64, 64, 1)
        self.register_buffer("temperature", torch.tensor(0.1), persistent=True)
        after_parent_rng = torch.get_rng_state()
        torch.set_rng_state(after_encoder_rng)
        self.guide_projection = nn.Conv2d(192, 128, 1)
        self.guide_interaction = PairInteractionBlock()
        self.guide_descriptor = nn.Linear(128, 128, bias=False)
        self.guide_null = nn.Linear(128, 1)
        self.guide_sigma = nn.Linear(128, 2)
        self.interaction = nn.ModuleList([AdaptiveSpanInteractionBlock(), AdaptiveSpanInteractionBlock()])
        torch.set_rng_state(after_parent_rng)
        self.branch_counts: dict[str, int] = {
            "encoder": 0,
            "interaction": 0,
            "guide": 0,
            "sampler": 0,
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
        coarse0, fine0, guide0 = self.encoder(image0)
        coarse1, fine1, guide1 = self.encoder(image1)
        self.branch_counts["encoder"] += 1
        return {"coarse0": coarse0, "coarse1": coarse1, "fine0": fine0, "fine1": fine1, "guide0": guide0, "guide1": guide1}

    @staticmethod
    def _guide_coordinates(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y, x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype), torch.arange(width, device=device, dtype=dtype), indexing="ij")
        return torch.stack((2.0 * x + 0.5, 2.0 * y + 0.5), dim=-1).reshape(-1, 2)

    def guide_routes(
        self,
        guide0: torch.Tensor,
        guide1: torch.Tensor,
        mask0: torch.Tensor,
        mask1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        projected0 = self.guide_projection(guide0)
        projected1 = self.guide_projection(guide1)
        h0, w0 = projected0.shape[-2:]
        h1, w1 = projected1.shape[-2:]
        token0 = projected0.flatten(2).transpose(1, 2)
        token1 = projected1.flatten(2).transpose(1, 2)
        flat_mask0 = mask0.flatten(1)
        flat_mask1 = mask1.flatten(1)
        token0, token1 = self.guide_interaction(token0, token1, flat_mask0, flat_mask1, (h0, w0), (h1, w1))
        descriptor0 = F.normalize(self.guide_descriptor(token0), dim=-1)
        descriptor1 = F.normalize(self.guide_descriptor(token1), dim=-1)
        logits = torch.einsum("bic,bjc->bij", descriptor0, descriptor1) / self.temperature
        valid = flat_mask0[:, :, None] & flat_mask1[:, None, :]
        logits = logits.float().masked_fill(~valid, -torch.inf)

        def categorical(query: torch.Tensor, pair_logits: torch.Tensor, query_mask: torch.Tensor, opposite_count: int) -> torch.Tensor:
            output = F.softmax(torch.cat((pair_logits, self.guide_null(query).float()), dim=2), dim=2)
            null_one = F.one_hot(torch.full_like(query_mask, opposite_count, dtype=torch.long), opposite_count + 1).to(output.dtype)
            return torch.where(query_mask[:, :, None], output, null_one)

        distribution01 = categorical(token0, logits, flat_mask0, token1.shape[1])
        distribution10 = categorical(token1, logits.transpose(1, 2), flat_mask1, token0.shape[1])

        def statistics(query: torch.Tensor, distribution: torch.Tensor, target_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            coordinates = self._guide_coordinates(*target_hw, query.device, query.dtype)
            non_null = distribution[:, :, :-1]
            covisibility = non_null.sum(dim=2, keepdim=True)
            normalized = non_null / covisibility.clamp_min(1e-8)
            mu = torch.einsum("bls,sd->bld", normalized, coordinates)
            fallback = coordinates.mean(dim=0)[None, None].expand_as(mu)
            mu = torch.where(covisibility > 0, mu, fallback)
            sigma = (F.softplus(self.guide_sigma(query).float()) + 0.25).clamp_max(12.0)
            return mu, sigma, covisibility

        mu01, sigma01, covis01 = statistics(token0, distribution01, (h1, w1))
        mu10, sigma10, covis10 = statistics(token1, distribution10, (h0, w0))
        self.branch_counts["guide"] += 1
        return {"guide_state0": token0, "guide_state1": token1, "guide_distribution01": distribution01, "guide_distribution10": distribution10, "guide_mu01": mu01, "guide_mu10": mu10, "guide_sigma01": sigma01, "guide_sigma10": sigma10, "guide_covis01": covis01, "guide_covis10": covis10, "guide_hw0": (h0, w0), "guide_hw1": (h1, w1)}

    @staticmethod
    def _upsample_route(value: torch.Tensor, source_hw: tuple[int, int], target_hw: tuple[int, int]) -> torch.Tensor:
        batch, _, channels = value.shape
        feature = value.transpose(1, 2).reshape(batch, channels, *source_hw)
        return F.interpolate(feature, size=target_hw, mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)

    def interact(
        self,
        coarse0: torch.Tensor,
        coarse1: torch.Tensor,
        mask0: torch.Tensor,
        mask1: torch.Tensor,
        routes: dict[str, torch.Tensor],
        cross_multiplier: float = 1.0,
        route_override: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        h0, w0 = coarse0.shape[-2:]
        h1, w1 = coarse1.shape[-2:]
        source = coarse0.flatten(2).transpose(1, 2)
        target = coarse1.flatten(2).transpose(1, 2)
        source_mask = mask0.flatten(1).bool()
        target_mask = mask1.flatten(1).bool()
        mu01 = self._upsample_route(routes["guide_mu01"], routes["guide_hw0"], (h0, w0))
        mu10 = self._upsample_route(routes["guide_mu10"], routes["guide_hw1"], (h1, w1))
        sigma01 = self._upsample_route(routes["guide_sigma01"], routes["guide_hw0"], (h0, w0))
        sigma10 = self._upsample_route(routes["guide_sigma10"], routes["guide_hw1"], (h1, w1))
        covis01 = self._upsample_route(routes["guide_covis01"], routes["guide_hw0"], (h0, w0))
        covis10 = self._upsample_route(routes["guide_covis10"], routes["guide_hw1"], (h1, w1))
        if route_override is not None:
            mu01 = route_override.get("mu01", mu01); mu10 = route_override.get("mu10", mu10)
        radius01 = (2.0 + 1.5 * sigma01).clamp(2.0, 12.0)
        radius10 = (2.0 + 1.5 * sigma10).clamp(2.0, 12.0)
        audits = {}
        for index, block in enumerate(self.interaction):
            source_self = block.self_update(source, source_mask, (h0, w0))
            target_self = block.self_update(target, target_mask, (h1, w1))
            source, audit01 = block.cross_update(
                source_self, target_self, source_mask, target_mask, (h1, w1),
                mu01, radius01, covis01, cross_multiplier,
            )
            target, audit10 = block.cross_update(
                target_self, source_self, target_mask, source_mask, (h0, w0),
                mu10, radius10, covis10, cross_multiplier,
            )
            audits[f"vasi_block{index}_message01"] = audit01["message"]
            audits[f"vasi_block{index}_message10"] = audit10["message"]
        self.branch_counts["interaction"] += len(self.interaction)
        self.branch_counts["sampler"] += 2 * len(self.interaction)
        audits.update({"route_mu01_s8": mu01, "route_mu10_s8": mu10, "route_sigma01_s8": sigma01, "route_sigma10_s8": sigma10, "route_radius01": radius01, "route_radius10": radius10, "route_covis01_s8": covis01, "route_covis10_s8": covis10})
        return source, target, audits

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
        cross_multiplier: float = 1.0,
        route_override: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        features = self.encode_pair(image0, image1)
        coarse_mask0 = self.coarse_center_mask(mask0, *features["coarse0"].shape[-2:])
        coarse_mask1 = self.coarse_center_mask(mask1, *features["coarse1"].shape[-2:])
        guide_mask0 = self.coarse_center_mask(mask0, *features["guide0"].shape[-2:], stride=16)
        guide_mask1 = self.coarse_center_mask(mask1, *features["guide1"].shape[-2:], stride=16)
        routes = self.guide_routes(features["guide0"], features["guide1"], guide_mask0, guide_mask1)
        source, target, interaction_audit = self.interact(
            features["coarse0"], features["coarse1"], coarse_mask0, coarse_mask1,
            routes, cross_multiplier=cross_multiplier, route_override=route_override,
        )
        assignment = self.assign(source, target, coarse_mask0.flatten(1), coarse_mask1.flatten(1))
        output: dict[str, Any] = {
            **features,
            "source_state": source,
            "target_state": target,
            "coarse_mask0": coarse_mask0,
            "coarse_mask1": coarse_mask1,
            "guide_mask0": guide_mask0,
            "guide_mask1": guide_mask1,
            **routes,
            **interaction_audit,
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
