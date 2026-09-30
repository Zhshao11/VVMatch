from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

from tools.prompt37_igm_hp1_common import PairSample, load_pair


def _matrix(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    angle = float(rng.uniform(-30.0, 30.0))
    scale = float(np.exp(rng.uniform(math.log(0.7), math.log(1.4))))
    center = (width / 2.0, height / 2.0)
    affine = cv2.getRotationMatrix2D(center, angle, scale)
    affine[:, 2] += [rng.uniform(-0.08, 0.08) * width, rng.uniform(-0.08, 0.08) * height]
    matrix = np.eye(3); matrix[:2] = affine
    jitter = 0.025
    matrix[2, 0] = rng.uniform(-jitter, jitter) / width
    matrix[2, 1] = rng.uniform(-jitter, jitter) / height
    return matrix


def _warp(tensor: torch.Tensor, mask: torch.Tensor, matrix: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    image = tensor[0, 0].numpy()
    valid = mask[0].numpy().astype(np.uint8)
    height, width = image.shape
    warped = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    warped_mask = cv2.warpPerspective(valid, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).astype(bool)
    return torch.from_numpy(warped)[None, None], torch.from_numpy(warped_mask)[None]


def load_gcrp_pair(root: Path, row: dict[str, str], seed: int, exposure: int) -> PairSample:
    sample = load_pair(root, row, seed, exposure, augment=True)
    rng = np.random.default_rng(seed + exposure * 1000003)
    height, width = sample.image0.shape[-2:]
    for _ in range(32):
        a0, a1 = _matrix(height, width, rng), _matrix(height, width, rng)
        image0, mask0 = _warp(sample.image0, sample.mask0, a0)
        image1, mask1 = _warp(sample.image1, sample.mask1, a1)
        overlap = min(float(mask0.float().mean()), float(mask1.float().mean()))
        if overlap >= 0.35:
            composed = torch.from_numpy((a1 @ sample.homography[0].numpy() @ np.linalg.inv(a0)).astype(np.float32))[None]
            metadata = {**sample.metadata, "gcrp_A0": a0.tolist(), "gcrp_A1": a1.tolist(), "gcrp_overlap": overlap}
            return replace(sample, image0=image0, image1=image1, mask0=mask0, mask1=mask1, homography=composed, metadata=metadata)
    raise RuntimeError("frozen GCRP transform stream could not reach overlap >=0.35")
