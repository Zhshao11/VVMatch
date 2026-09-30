from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from tools.igm_m0r1.model import IndependentPyramid


class GCRPPretrainer(nn.Module):
    """Wraps the exact M0R1 encoder with training-only representation heads."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = IndependentPyramid()
        self.ema_encoder = copy.deepcopy(self.encoder)
        for parameter in self.ema_encoder.parameters():
            parameter.requires_grad_(False)
        self.fine_adapter = nn.Linear(64, 128)
        self.guide_adapter = nn.Identity()
        self.projector = nn.Linear(128, 128)
        self.predictor = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))
        self.scale_embedding = nn.Parameter(torch.zeros(3, 128))
        self.branch_counts = {"mask": 0, "predictor": 0, "ema": 0, "contrast": 0, "geometry": 0, "variance_covariance": 0}

    @staticmethod
    def _features(encoder: IndependentPyramid, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x2 = encoder.stage2(encoder.stem(image))
        x3 = encoder.stage3(x2)
        x4 = encoder.stage4(x3)
        guide = encoder.lateral4(x4)
        coarse = encoder.coarse_out(encoder.lateral3(x3) + F.interpolate(guide, size=x3.shape[-2:], mode="bilinear", align_corners=False))
        fine = encoder.fine_out(encoder.lateral2(x2) + F.interpolate(encoder.coarse_to_fine(coarse), size=x2.shape[-2:], mode="bilinear", align_corners=False))
        return {"fine": fine, "coarse": coarse, "guide": guide}

    def project(self, feature: torch.Tensor, scale_index: int) -> torch.Tensor:
        tokens = feature.flatten(2).transpose(1, 2)
        if scale_index == 0:
            tokens = self.fine_adapter(tokens)
        tokens = tokens + self.scale_embedding[scale_index]
        return F.normalize(self.projector(tokens), dim=-1)

    def forward(self, image0: torch.Tensor, image1: torch.Tensor, mask0: torch.Tensor, mask1: torch.Tensor) -> dict[str, Any]:
        pixel_mask0 = F.interpolate(mask0[:, None].float(), size=image0.shape[-2:], mode="nearest").bool()
        pixel_mask1 = F.interpolate(mask1[:, None].float(), size=image1.shape[-2:], mode="nearest").bool()
        student0, student1 = self._features(self.encoder, image0.masked_fill(pixel_mask0, 0)), self._features(self.encoder, image1.masked_fill(pixel_mask1, 0))
        with torch.no_grad():
            target0, target1 = self._features(self.ema_encoder, image0), self._features(self.ema_encoder, image1)
        output: dict[str, Any] = {"student_features0": student0, "student_features1": student1, "target_features0": target0, "target_features1": target1, "token_mask0": mask0, "token_mask1": mask1}
        for index, name in enumerate(("fine", "coarse", "guide")):
            output[f"student_{name}0"] = self.project(student0[name], index)
            output[f"student_{name}1"] = self.project(student1[name], index)
            with torch.no_grad():
                output[f"target_{name}0"] = self.project(target0[name], index).detach()
                output[f"target_{name}1"] = self.project(target1[name], index).detach()
        self.branch_counts["ema"] += 1
        return output

    @torch.no_grad()
    def update_ema(self, decay: float = 0.996) -> None:
        for target, source in zip(self.ema_encoder.parameters(), self.encoder.parameters()):
            target.mul_(decay).add_(source, alpha=1.0 - decay)

    def export_encoder(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in self.encoder.state_dict().items()}
