"""Isolated scratch-training adapter for the clean public baseline source."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from tools.prompt37_igm_hp1_common import PairSample, warp_points


class OfficialScratchAdapter:
    def __init__(self, source_root: Path, seed: int, device: torch.device) -> None:
        self.source_root = source_root.resolve()
        if not (self.source_root / "src/slim.py").is_file():
            raise FileNotFoundError(self.source_root / "src/slim.py")
        if str(self.source_root) not in sys.path:
            sys.path.insert(0, str(self.source_root))
        get_config = importlib.import_module("default_config").get_config
        model_class = importlib.import_module("src.slim").SLiM
        loss_class = importlib.import_module("src.loss").SLiM_Loss
        config = get_config("outdoor_train")
        config.AMP = False
        config.BATCH_SIZE = 1
        config.MODEL.MAX_COARSE_MATCHES = 6000
        config.MODEL.MAX_FINE_MATCHES = 6000
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.model = model_class(config.MODEL).to(device)
        self.loss_module = loss_class(config.LOSS).to(device)
        self.config = config
        self.device = device
        self.loaded_checkpoint = False

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state, strict=True)

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    @staticmethod
    def _centers(height: int, width: int, device: torch.device) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
        )
        return torch.stack(((x + 0.5) * 8, (y + 0.5) * 8), dim=-1).reshape(-1, 2)

    @staticmethod
    def _coarse_mask(pixel_mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
        valid_height = pixel_mask.any(dim=2).sum(dim=1)
        valid_width = pixel_mask.any(dim=1).sum(dim=1)
        center_y = (torch.arange(height, device=pixel_mask.device) + 0.5) * 8
        center_x = (torch.arange(width, device=pixel_mask.device) + 0.5) * 8
        return (center_y[None, :, None] < valid_height[:, None, None]) & (
            center_x[None, None, :] < valid_width[:, None, None]
        )

    def _supervision(self, sample: PairSample) -> dict[str, torch.Tensor]:
        h0, w0 = sample.image0.shape[-2] // 8, sample.image0.shape[-1] // 8
        h1, w1 = sample.image1.shape[-2] // 8, sample.image1.shape[-1] // 8
        mask0 = self._coarse_mask(sample.mask0, h0, w0)
        mask1 = self._coarse_mask(sample.mask1, h1, w1)
        center0 = self._centers(h0, w0, self.device)
        center1 = self._centers(h1, w1, self.device)
        projected1 = warp_points(center0[None], sample.homography)[0]
        x1 = torch.round(projected1[:, 0] / 8 - 0.5).long()
        y1 = torch.round(projected1[:, 1] / 8 - 0.5).long()
        inside1 = (x1 >= 0) & (x1 < w1) & (y1 >= 0) & (y1 < h1)
        safe_j = y1.clamp(0, h1 - 1) * w1 + x1.clamp(0, w1 - 1)
        projected0 = warp_points(center1[None], torch.linalg.inv(sample.homography))[0]
        x0 = torch.round(projected0[:, 0] / 8 - 0.5).long()
        y0 = torch.round(projected0[:, 1] / 8 - 0.5).long()
        safe_i_back = y0.clamp(0, h0 - 1) * w0 + x0.clamp(0, w0 - 1)
        all_i = torch.arange(h0 * w0, device=self.device)
        mutual = safe_i_back[safe_j] == all_i
        valid = mask0.flatten()[all_i] & inside1 & mask1.flatten()[safe_j] & mutual
        i_ids = all_i[valid]
        j_ids = safe_j[valid]
        if i_ids.numel() == 0:
            raise RuntimeError("baseline supervision has no valid coarse matches")
        b_ids = torch.zeros_like(i_ids)
        conf_gt = torch.zeros(1, h0 * w0, h1 * w1, device=self.device)
        conf_gt[0, i_ids, j_ids] = 1.0

        coarse0 = torch.stack(((i_ids % w0) * 8, (i_ids // w0) * 8), dim=1).float()
        coarse1 = torch.stack(((j_ids % w1) * 8, (j_ids // w1) * 8), dim=1).float()
        source_fine = coarse0 + 4.5
        target_exact = warp_points(source_fine[None], sample.homography)[0]
        relative1 = torch.round(target_exact - coarse1 - 0.5).long()
        fine_valid = (relative1 >= 0).all(dim=1) & (relative1 < 8).all(dim=1)
        chosen = torch.where(fine_valid)[0]
        if chosen.numel() == 0:
            raise RuntimeError("baseline supervision has no valid fine matches")
        spv_b = b_ids[chosen]
        spv_m = chosen
        spv_i = torch.full_like(chosen, 4 * 8 + 4)
        spv_j = relative1[chosen, 1] * 8 + relative1[chosen, 0]
        intermediate0 = source_fine[chosen]
        intermediate1 = coarse1[chosen] + relative1[chosen].float() + 0.5
        conf_f_gt = torch.zeros(i_ids.numel(), 64, 64, device=self.device)
        conf_f_gt[chosen, spv_i, spv_j] = 1.0
        return {
            "mask0": mask0,
            "mask1": mask1,
            "scale0": torch.ones(1, 2, device=self.device),
            "scale1": torch.ones(1, 2, device=self.device),
            "conf_matrix_gt": conf_gt,
            "spv_b_ids": b_ids,
            "spv_i_ids": i_ids,
            "spv_j_ids": j_ids,
            "conf_matrix_f_gt": conf_f_gt,
            "spv_b_ids_it": spv_b,
            "spv_m_ids_it": spv_m,
            "spv_i_ids_it": spv_i,
            "spv_j_ids_it": spv_j,
            "intermediate_coord_0_gt": intermediate0,
            "intermediate_coord_1_gt": intermediate1,
        }

    def training_loss(self, sample: PairSample) -> tuple[torch.Tensor, dict[str, float]]:
        data: dict[str, Any] = {
            "image0": sample.image0,
            "image1": sample.image1,
            **self._supervision(sample),
        }
        self.model(data, training=True)
        target = warp_points(data["fine_coord_0"][None], sample.homography)[0]
        offset = (target - data["intermediate_coord_1"]) / 3.0
        correct = torch.linalg.vector_norm(offset, ord=float("inf"), dim=1) < 2.0
        data["coord_offset_gt"] = offset
        data["correct_mask"] = correct
        self.loss_module(data)
        scalars = {key: float(value) for key, value in data["loss_scalars"].items()}
        scalars["loss_total"] = float(data["loss"].detach())
        scalars["coarse_matches_gt"] = float(data["spv_i_ids"].numel())
        scalars["fine_matches_gt"] = float(data["spv_i_ids_it"].numel())
        return data["loss"], scalars

    @torch.no_grad()
    def infer(self, sample: PairSample) -> dict[str, torch.Tensor]:
        h0, w0 = sample.image0.shape[-2] // 8, sample.image0.shape[-1] // 8
        h1, w1 = sample.image1.shape[-2] // 8, sample.image1.shape[-1] // 8
        data: dict[str, Any] = {
            "image0": sample.image0,
            "image1": sample.image1,
            "mask0": self._coarse_mask(sample.mask0, h0, w0),
            "mask1": self._coarse_mask(sample.mask1, h1, w1),
            "scale0": torch.ones(1, 2, device=self.device),
            "scale1": torch.ones(1, 2, device=self.device),
        }
        self.model(data, training=False)
        if data["fine_coord_0"].numel() == 0:
            return {
                "source_xy": data["fine_coord_0"],
                "target_xy": data["fine_coord_1"],
                "confidence": data["fine_coord_0"].new_zeros((0,)),
            }
        coarse_conf = F.softmax(data["sim_matrix"], 1) * F.softmax(data["sim_matrix"], 2)
        coarse_value = coarse_conf[data["b_idx_c"], data["i_idx_c"], data["j_idx_c"]]
        fine_conf = F.softmax(data["sim_matrix_f"], 1) * F.softmax(data["sim_matrix_f"], 2)
        fine_value = fine_conf[data["m_idx_it"], data["i_idx_it"], data["j_idx_it"]]
        confidence = coarse_value[data["m_idx_it"]] * fine_value
        return {
            "source_xy": data["fine_coord_0"],
            "target_xy": data["fine_coord_1"],
            "confidence": confidence,
        }
