from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def _seed64(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def read_pair_manifest(path: Path, role: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["role"] == role]
    if not rows:
        raise ValueError(f"no {role} rows in {path}")
    return rows


def deterministic_pair(rows: list[dict[str, str]], seed: int, exposure: int) -> dict[str, str]:
    epoch, offset = divmod(exposure, len(rows))
    order = np.random.default_rng(_seed64("pair-order", seed, epoch)).permutation(len(rows))
    return rows[int(order[offset])]


def _transform_matrix(
    original_hw: tuple[int, int], crop_fraction: float, offset_x: float, offset_y: float, flip: bool
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = original_hw
    crop_h = max(32, min(height, int(round(height * crop_fraction))))
    crop_w = max(32, min(width, int(round(width * crop_fraction))))
    left = int(round((width - crop_w) * offset_x))
    top = int(round((height - crop_h) * offset_y))
    crop = np.array([[1.0, 0.0, -left], [0.0, 1.0, -top], [0.0, 0.0, 1.0]])
    if flip:
        mirror = np.array([[-1.0, 0.0, float(crop_w)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        crop = mirror @ crop
    scale = 480.0 / min(crop_h, crop_w)
    resized_h = int(round(crop_h * scale))
    resized_w = int(round(crop_w * scale))
    resize = np.array([[resized_w / crop_w, 0.0, 0.0], [0.0, resized_h / crop_h, 0.0], [0.0, 0.0, 1.0]])
    return resize @ crop, (left, top, crop_w, crop_h)


def _apply_transform(
    image: np.ndarray,
    matrix: np.ndarray,
    crop: tuple[int, int, int, int],
    flip: bool,
) -> np.ndarray:
    left, top, crop_w, crop_h = crop
    image = image[top : top + crop_h, left : left + crop_w]
    if flip:
        image = image[:, ::-1]
    scale_x = matrix[0, 0] if not flip else -matrix[0, 0]
    resized_w = int(round(crop_w * abs(scale_x)))
    resized_h = int(round(crop_h * matrix[1, 1]))
    return cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)


def _photometric(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]:
    brightness = float(rng.uniform(-0.08, 0.08))
    contrast = float(rng.uniform(0.85, 1.15))
    gamma = float(rng.uniform(0.85, 1.15))
    noise = float(rng.uniform(0.0, 0.02))
    blur = int(rng.choice([0, 0, 0, 3, 5]))
    value = np.clip(image.astype(np.float32) / 255.0 * contrast + brightness, 0.0, 1.0)
    value = np.power(value, gamma)
    if noise > 0:
        value = value + rng.normal(0.0, noise, size=value.shape).astype(np.float32)
    value = np.clip(value, 0.0, 1.0)
    if blur:
        value = cv2.GaussianBlur(value, (blur, blur), 0)
    return value.astype(np.float32), {
        "brightness": brightness,
        "contrast": contrast,
        "gamma": gamma,
        "noise_sigma": noise,
        "blur_kernel": blur,
    }


def _pad(image: np.ndarray, multiple: int = 16) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    padded_h = int(math.ceil(height / multiple) * multiple)
    padded_w = int(math.ceil(width / multiple) * multiple)
    output = np.zeros((padded_h, padded_w), dtype=np.float32)
    mask = np.zeros((padded_h, padded_w), dtype=bool)
    output[:height, :width] = image
    mask[:height, :width] = True
    return output, mask


def _pad_pair_common(
    source: np.ndarray, target: np.ndarray, multiple: int = 16
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    padded_h = int(math.ceil(max(source.shape[0], target.shape[0]) / multiple) * multiple)
    padded_w = int(math.ceil(max(source.shape[1], target.shape[1]) / multiple) * multiple)
    outputs = []
    for image in (source, target):
        output = np.zeros((padded_h, padded_w), dtype=np.float32)
        mask = np.zeros((padded_h, padded_w), dtype=bool)
        output[: image.shape[0], : image.shape[1]] = image
        mask[: image.shape[0], : image.shape[1]] = True
        outputs.extend((output, mask))
    return outputs[0], outputs[1], outputs[2], outputs[3]


@dataclass
class PairSample:
    image0: torch.Tensor
    image1: torch.Tensor
    mask0: torch.Tensor
    mask1: torch.Tensor
    homography: torch.Tensor
    original_homography: torch.Tensor
    original_to_network0: torch.Tensor
    original_to_network1: torch.Tensor
    valid_hw0: tuple[int, int]
    valid_hw1: tuple[int, int]
    metadata: dict[str, Any]


def load_pair(
    root: Path,
    row: dict[str, str],
    seed: int,
    exposure: int,
    augment: bool,
) -> PairSample:
    source = cv2.imread(str(root / row["source_image"]), cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(str(root / row["target_image"]), cv2.IMREAD_GRAYSCALE)
    if source is None or target is None:
        raise FileNotFoundError(f"missing pair {row['sample_id']}")
    homography = np.loadtxt(root / row["homography"], dtype=np.float64).reshape(3, 3)
    if row["invert_homography"].lower() == "true":
        homography = np.linalg.inv(homography)
    original_homography = homography.copy()
    rng = np.random.default_rng(_seed64("augmentation", seed, exposure, row["sample_id"]))
    rng_state_before = rng.bit_generator.state
    if augment:
        crop_fraction = float(rng.uniform(0.90, 1.0))
        offset_x = float(rng.uniform(0.0, 1.0))
        offset_y = float(rng.uniform(0.0, 1.0))
        flip = bool(rng.integers(0, 2))
    else:
        crop_fraction, offset_x, offset_y, flip = 1.0, 0.0, 0.0, False
    transform0, crop0 = _transform_matrix(source.shape, crop_fraction, offset_x, offset_y, flip)
    transform1, crop1 = _transform_matrix(target.shape, crop_fraction, offset_x, offset_y, flip)
    source = _apply_transform(source, transform0, crop0, flip)
    target = _apply_transform(target, transform1, crop1, flip)
    if augment:
        source, photo0 = _photometric(source, rng)
        target, photo1 = _photometric(target, rng)
    else:
        source, target = source.astype(np.float32) / 255.0, target.astype(np.float32) / 255.0
        photo0 = photo1 = {"brightness": 0.0, "contrast": 1.0, "gamma": 1.0, "noise_sigma": 0.0, "blur_kernel": 0}
    valid_hw0, valid_hw1 = source.shape, target.shape
    source, mask0, target, mask1 = _pad_pair_common(source, target)
    network_h = transform1 @ homography @ np.linalg.inv(transform0)
    metadata = {
        **row,
        "seed": seed,
        "exposure": exposure,
        "rng_seed": _seed64("augmentation", seed, exposure, row["sample_id"]),
        "rng_state_before": rng_state_before,
        "crop_fraction": crop_fraction,
        "crop_offset_x": offset_x,
        "crop_offset_y": offset_y,
        "flip": flip,
        "photo0": photo0,
        "photo1": photo1,
        "valid_hw0": list(valid_hw0),
        "valid_hw1": list(valid_hw1),
    }
    return PairSample(
        image0=torch.from_numpy(source)[None, None],
        image1=torch.from_numpy(target)[None, None],
        mask0=torch.from_numpy(mask0)[None],
        mask1=torch.from_numpy(mask1)[None],
        homography=torch.from_numpy(network_h.astype(np.float32))[None],
        original_homography=torch.from_numpy(original_homography.astype(np.float32))[None],
        original_to_network0=torch.from_numpy(transform0.astype(np.float32))[None],
        original_to_network1=torch.from_numpy(transform1.astype(np.float32))[None],
        valid_hw0=valid_hw0,
        valid_hw1=valid_hw1,
        metadata=metadata,
    )


def move_sample(sample: PairSample, device: torch.device) -> PairSample:
    values = vars(sample).copy()
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            values[key] = value.to(device, non_blocking=True)
    return PairSample(**values)


def warp_points(points: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(points[..., :1])
    homogeneous = torch.cat((points, ones), dim=-1)
    warped = torch.einsum("bij,bnj->bni", homography, homogeneous)
    denominator = warped[..., 2:]
    denominator = torch.where(
        denominator.abs() < 1e-8,
        torch.where(denominator < 0, -torch.ones_like(denominator), torch.ones_like(denominator)) * 1e-8,
        denominator,
    )
    return warped[..., :2] / denominator


def restore_points(points: torch.Tensor, original_to_network: torch.Tensor) -> torch.Tensor:
    inverse = torch.linalg.inv(original_to_network)
    return warp_points(points[None] if points.ndim == 2 else points, inverse)[0]


def metadata_json(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))
