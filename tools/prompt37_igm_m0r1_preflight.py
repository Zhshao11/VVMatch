#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.igm_m0 import IndependentGeometryMatcherM0
from tools.igm_m0r1 import IndependentGeometryMatcherM0R1
from tools.igm_m0r1.losses import compute_igm_loss
from tools.prompt37_igm_hp1_common import PairSample, warp_points


EXPERIMENT = "P37-20260924-IGM-M0R1-HP1"


def sha_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def synthetic_image(size: int = 64) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    value = (
        0.35 * np.sin(x * 0.31)
        + 0.25 * np.cos(y * 0.23)
        + 0.20 * np.sin((x + y) * 0.17)
        + ((x // 8 + y // 8) % 2) * 0.35
    )
    value = (value - value.min()) / (value.max() - value.min())
    return value.astype(np.float32)


def make_sample(image: np.ndarray, homography: np.ndarray, name: str, device: torch.device) -> PairSample:
    height, width = image.shape
    target = cv2.warpPerspective(image, homography, (width, height), flags=cv2.INTER_LINEAR)
    mask = torch.ones(1, height, width, dtype=torch.bool, device=device)
    identity = torch.eye(3, dtype=torch.float32, device=device)[None]
    return PairSample(
        image0=torch.from_numpy(image)[None, None].to(device),
        image1=torch.from_numpy(target)[None, None].to(device),
        mask0=mask,
        mask1=mask,
        homography=torch.from_numpy(homography.astype(np.float32))[None].to(device),
        original_homography=torch.from_numpy(homography.astype(np.float32))[None].to(device),
        original_to_network0=identity,
        original_to_network1=identity,
        valid_hw0=(height, width),
        valid_hw1=(height, width),
        metadata={"sample_id": name},
    )


def free_logit_control(device: torch.device) -> dict:
    model = IndependentGeometryMatcherM0R1().to(device)
    source_mask = torch.tensor([[True, True, True, True, False]], device=device)
    target_mask = torch.tensor([[True, True, True, True, False]], device=device)
    cases = {
        "identity": ([0, 1, 2, 3], [0, 1, 2, 3], [], []),
        "translation": ([0, 1, 2], [1, 2, 3], [3], [0]),
        "repeated_texture": ([0, 1], [1, 0], [2, 3], [2, 3]),
        "unmatched": ([0, 1], [0, 1], [2, 3], [2, 3]),
        "padded": ([0, 1, 2, 3], [0, 1, 2, 3], [], []),
    }
    rows = []
    for case, (source_ids, target_ids, source_null_ids, target_null_ids) in cases.items():
        logits = torch.zeros(1, 5, 5, device=device, requires_grad=True)
        source_null = torch.zeros(1, 5, 1, device=device, requires_grad=True)
        target_null = torch.zeros(1, 5, 1, device=device, requires_grad=True)
        optimizer = torch.optim.Adam((logits, source_null, target_null), lr=0.3)
        for _ in range(120):
            output = model.categorical_assignment(logits, source_null, target_null, source_mask, target_mask)
            loss = torch.zeros((), device=device)
            for source_id, target_id in zip(source_ids, target_ids):
                loss -= torch.log(output["source_assignment"][0, source_id, target_id].clamp_min(1e-8))
                loss -= torch.log(output["target_assignment"][0, target_id, source_id].clamp_min(1e-8))
            for source_id in source_null_ids:
                loss -= torch.log(output["source_assignment"][0, source_id, -1].clamp_min(1e-8))
            for target_id in target_null_ids:
                loss -= torch.log(output["target_assignment"][0, target_id, -1].clamp_min(1e-8))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        output = model.categorical_assignment(logits, source_null, target_null, source_mask, target_mask)
        _, recovered_source, recovered_target, _ = model.mutual_matches(output["source_assignment"], output["target_assignment"])
        expected = list(zip(source_ids, target_ids))
        recovered = list(zip(recovered_source.tolist(), recovered_target.tolist()))
        source_null_ok = all(int(output["source_assignment"][0, index].argmax()) == 5 for index in source_null_ids)
        target_null_ok = all(int(output["target_assignment"][0, index].argmax()) == 5 for index in target_null_ids)
        padded_ok = int(output["source_assignment"][0, 4].argmax()) == 5 and int(output["target_assignment"][0, 4].argmax()) == 5
        rows.append({"case": case, "expected": expected, "recovered": recovered, "source_null_ok": source_null_ok, "target_null_ok": target_null_ok, "padded_ok": padded_ok, "passed": recovered == expected and source_null_ok and target_null_ok and padded_ok})
    return {"cases": rows, "passed": all(row["passed"] for row in rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    device = torch.device("cuda")
    torch.manual_seed(20260923); torch.cuda.manual_seed_all(20260923)

    torch.manual_seed(20260923)
    parent = IndependentGeometryMatcherM0().to(device)
    torch.manual_seed(20260923)
    rescue = IndependentGeometryMatcherM0R1().to(device)
    parent_rows = []
    for key, value in parent.state_dict().items():
        same = torch.equal(value, rescue.state_dict()[key])
        parent_rows.append({"key": key, "parent_sha256": sha_tensor(value), "rescue_sha256": sha_tensor(rescue.state_dict()[key]), "exact": same})
    (args.output / "parent_initialization_parity.json").write_text(json.dumps(parent_rows, indent=2) + "\n")
    if not all(row["exact"] for row in parent_rows):
        raise RuntimeError("parent initialization parity failed")

    free = free_logit_control(device)
    (args.output / "free_logit_control.json").write_text(json.dumps(free, indent=2) + "\n")
    if not free["passed"]:
        raise RuntimeError(f"free-logit control failed: {free}")

    image = synthetic_image()
    homographies = {
        "identity": np.eye(3, dtype=np.float64),
        "translation": np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]),
        "scale": np.array([[1.04, 0.0, -1.28], [0.0, 1.04, -1.28], [0.0, 0.0, 1.0]]),
    }
    samples = [make_sample(image, homography, name, device) for name, homography in homographies.items()]
    model = IndependentGeometryMatcherM0R1().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    trajectory = []
    for step in range(args.steps):
        sample = samples[step % len(samples)]
        output = model(sample.image0, sample.image1, sample.mask0, sample.mask1, return_matches=False)
        loss, metrics = compute_igm_loss(model, output, sample)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step % 20 == 0 or step + 1 == args.steps:
            trajectory.append({"step": step + 1, "case": sample.metadata["sample_id"], **metrics})
    (args.output / "student_synthetic_trajectory.json").write_text(json.dumps(trajectory, indent=2) + "\n")

    case_rows = []
    all_errors = []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            output = model(sample.image0, sample.image1, sample.mask0, sample.mask1, return_matches=True)
            matches = output["matches"]
            target_gt = warp_points(matches.source_xy[None], sample.homography)[0] if matches.source_xy.numel() else matches.target_xy
            errors = torch.linalg.vector_norm(target_gt - matches.target_xy, dim=1)
            all_errors.extend(errors.cpu().tolist())
            case_rows.append({
                "case": sample.metadata["sample_id"],
                "match_count": int(matches.source_xy.shape[0]),
                "median_transfer_px": float(errors.median()) if errors.numel() else float("inf"),
                "source_null_argmax_rate": float((output["source_assignment"].argmax(-1) == output["source_assignment"].shape[-1] - 1).float().mean()),
                "target_null_argmax_rate": float((output["target_assignment"].argmax(-1) == output["target_assignment"].shape[-1] - 1).float().mean()),
            })
    median = float(np.median(all_errors)) if all_errors else float("inf")
    gate = free["passed"] and median <= 0.5 and all(row["match_count"] > 0 for row in case_rows)
    result = {
        "experiment": EXPERIMENT,
        "host": socket.gethostname(),
        "steps": args.steps,
        "free_logit_passed": free["passed"],
        "parent_initialization_exact": all(row["exact"] for row in parent_rows),
        "student_cases": case_rows,
        "student_median_transfer_px": median,
        "student_gate_le_0_5": median <= 0.5,
        "all_cases_nonempty": all(row["match_count"] > 0 for row in case_rows),
        "passed": gate,
    }
    (args.output / "PREFLIGHT_SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not gate:
        raise RuntimeError("M0R1 positive-control preflight failed")


if __name__ == "__main__":
    main()
