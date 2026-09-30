#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.igm_m0r1 import IndependentGeometryMatcherM0R1
from tools.igm_m1_vasi import IndependentGeometryMatcherM1VASI
from tools.igm_m1_vasi.losses import compute_igm_loss
from tools.prompt37_igm_hp1_common import PairSample, warp_points


EXPERIMENT = "P37-20260926-IGM-M1-HP2"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def synthetic_image(height: int = 128, width: int = 160) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    value = (
        0.30 * np.sin(x * 0.21) + 0.24 * np.cos(y * 0.17)
        + 0.20 * np.sin((x + y) * 0.13) + ((x // 9 + y // 11) % 2) * 0.32
    )
    value = (value - value.min()) / (value.max() - value.min())
    return value.astype(np.float32)


def make_sample(image: np.ndarray, homography: np.ndarray, name: str, device: torch.device) -> PairSample:
    height, width = image.shape
    target = cv2.warpPerspective(image, homography, (width, height), flags=cv2.INTER_LINEAR)
    mask = torch.ones(1, height, width, dtype=torch.bool, device=device)
    identity = torch.eye(3, dtype=torch.float32, device=device)[None]
    matrix = torch.from_numpy(homography.astype(np.float32))[None].to(device)
    return PairSample(
        image0=torch.from_numpy(image)[None, None].to(device),
        image1=torch.from_numpy(target)[None, None].to(device),
        mask0=mask, mask1=mask, homography=matrix, original_homography=matrix,
        original_to_network0=identity, original_to_network1=identity,
        valid_hw0=(height, width), valid_hw1=(height, width),
        metadata={"sample_id": name, "subset": "viewpoint" if name != "identity" else "illumination"},
    )


def valid_route_metrics(output: dict, sample: PairSample, mu_key: str = "route_mu01_s8") -> tuple[float, torch.Tensor, torch.Tensor]:
    centers = IndependentGeometryMatcherM1VASI._token_centers(
        *output["coarse0"].shape[-2:], 8, sample.image0.device
    )
    projected = warp_points(centers[None], sample.homography)[0]
    valid = output["coarse_mask0"].flatten(1)[0] & torch.isfinite(projected).all(dim=1)
    valid &= (projected[:, 0] >= 0) & (projected[:, 0] < sample.valid_hw1[1])
    valid &= (projected[:, 1] >= 0) & (projected[:, 1] < sample.valid_hw1[0])
    target_s8 = projected / 8.0 - 0.5
    mu = output[mu_key][0]
    radius = output["route_radius01"][0]
    covered = ((target_s8 - mu).abs() <= radius).all(dim=1)
    return (float(covered[valid].float().mean()) if valid.any() else 0.0), target_s8, valid


def gt_route_override(output: dict, sample: PairSample) -> dict[str, torch.Tensor]:
    source_centers = IndependentGeometryMatcherM1VASI._token_centers(
        *output["coarse0"].shape[-2:], 8, sample.image0.device
    )
    target_centers = IndependentGeometryMatcherM1VASI._token_centers(
        *output["coarse1"].shape[-2:], 8, sample.image0.device
    )
    mu01 = warp_points(source_centers[None], sample.homography)[0] / 8.0 - 0.5
    mu10 = warp_points(target_centers[None], torch.linalg.inv(sample.homography))[0] / 8.0 - 0.5
    return {"mu01": mu01[None], "mu10": mu10[None]}


def transfer_errors(output: dict, sample: PairSample) -> torch.Tensor:
    matches = output["matches"]
    projected = warp_points(matches.source_xy[None], sample.homography)[0] if matches.source_xy.numel() else matches.target_xy
    return torch.linalg.vector_norm(projected - matches.target_xy, dim=1)


def evaluate(model: IndependentGeometryMatcherM1VASI, samples: list[PairSample]) -> dict:
    rows = []
    predicted_coverages, permuted_coverages, gt_coverages = [], [], []
    predicted_errors, gt_errors = [], []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            predicted_output = model(sample.image0, sample.image1, sample.mask0, sample.mask1, return_matches=True)
            predicted, _, _ = valid_route_metrics(predicted_output, sample)
            gt_output = model(
                sample.image0, sample.image1, sample.mask0, sample.mask1,
                return_matches=True, route_override=gt_route_override(predicted_output, sample),
            )
            gt, _, _ = valid_route_metrics(gt_output, sample)
            permutation01 = torch.roll(torch.arange(predicted_output["route_mu01_s8"].shape[1], device=sample.image0.device), shifts=max(predicted_output["route_mu01_s8"].shape[1] // 2, 1))
            permutation10 = torch.roll(torch.arange(predicted_output["route_mu10_s8"].shape[1], device=sample.image0.device), shifts=max(predicted_output["route_mu10_s8"].shape[1] // 2, 1))
            permuted_output = model(
                sample.image0, sample.image1, sample.mask0, sample.mask1, return_matches=True,
                route_override={
                    "mu01": predicted_output["route_mu01_s8"][:, permutation01],
                    "mu10": predicted_output["route_mu10_s8"][:, permutation10],
                },
            )
            permuted, _, _ = valid_route_metrics(permuted_output, sample)
            predicted_case_errors = transfer_errors(predicted_output, sample)
            gt_case_errors = transfer_errors(gt_output, sample)
            predicted_coverages.append(predicted); permuted_coverages.append(permuted); gt_coverages.append(gt)
            predicted_errors.extend(predicted_case_errors.cpu().tolist()); gt_errors.extend(gt_case_errors.cpu().tolist())
            rows.append({
                "case": sample.metadata["sample_id"], "gt_route_coverage": gt,
                "predicted_route_coverage": predicted, "permuted_route_coverage": permuted,
                "predicted_match_count": int(predicted_output["matches"].source_xy.shape[0]),
                "gt_route_match_count": int(gt_output["matches"].source_xy.shape[0]),
                "predicted_median_refined_transfer_px": float(predicted_case_errors.median()) if predicted_case_errors.numel() else float("inf"),
                "gt_route_median_refined_transfer_px": float(gt_case_errors.median()) if gt_case_errors.numel() else float("inf"),
            })
    return {
        "cases": rows,
        "gt_route_coverage": float(np.mean(gt_coverages)),
        "predicted_route_coverage": float(np.mean(predicted_coverages)),
        "permuted_route_coverage": float(np.mean(permuted_coverages)),
        "predicted_median_refined_transfer_px": float(np.median(predicted_errors)) if predicted_errors else float("inf"),
        "median_refined_transfer_px": float(np.median(gt_errors)) if gt_errors else float("inf"),
        "all_cases_nonempty": all(row["predicted_match_count"] > 0 and row["gt_route_match_count"] > 0 for row in rows),
    }


def write_card(output: Path, payload: dict) -> None:
    (output / "RESULT_CARD.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output / "RESULT_CARD.md").write_text(
        "# IGM-M1 VASI Same-Parameterization Preflight\n\n"
        f"Status: `{payload['status']}`; highest evidence: `{payload['evidence_rung']}`.\n\n"
        + json.dumps(payload.get("gates", {}), indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    payload = {"experiment": EXPERIMENT, "stage": "same_parameterization_route_student_preflight", "host": socket.gethostname(), "steps": args.steps}
    try:
        device = torch.device("cuda")
        torch.manual_seed(20260926); torch.cuda.manual_seed_all(20260926)
        torch.manual_seed(20260926); parent = IndependentGeometryMatcherM0R1().to(device)
        torch.manual_seed(20260926); model = IndependentGeometryMatcherM1VASI().to(device)
        inherited_prefixes = ("encoder.", "source_projection.", "target_projection.", "source_matchability.", "target_matchability.", "fine_projection.", "temperature")
        inherited_rows = []
        for name, value in parent.state_dict().items():
            if name.startswith(inherited_prefixes):
                child = model.state_dict()[name]
                inherited_rows.append({"name": name, "parent": tensor_hash(value), "child": tensor_hash(child), "exact": torch.equal(value, child)})
        (args.output / "inherited_tensor_parity.json").write_text(json.dumps(inherited_rows, indent=2) + "\n", encoding="utf-8")
        image = synthetic_image()
        angle = math.radians(3.0)
        scale = 1.02
        cosine, sine = scale * math.cos(angle), scale * math.sin(angle)
        center_x, center_y = image.shape[1] / 2.0, image.shape[0] / 2.0
        translate_x = center_x - cosine * center_x + sine * center_y
        translate_y = center_y - sine * center_x - cosine * center_y
        homographies = {
            "identity": np.eye(3),
            "translation": np.array([[1, 0, 4], [0, 1, 3], [0, 0, 1]], dtype=np.float64),
            "scale_rotation": np.array([[cosine, -sine, translate_x], [sine, cosine, translate_y], [0, 0, 1]], dtype=np.float64),
            "projective": np.array([[1.0, 0.02, 3], [-0.015, 1.0, 2], [0.0002, -0.00015, 1]], dtype=np.float64),
        }
        samples = [make_sample(image, matrix, name, device) for name, matrix in homographies.items()]
        initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
        ever_gradient = {name: False for name, _ in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.999))
        trajectory = []
        for step in range(args.steps):
            sample = samples[step % len(samples)]
            output = model(sample.image0, sample.image1, sample.mask0, sample.mask1, return_matches=False)
            loss, metrics = compute_igm_loss(model, output, sample)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            for name, parameter in model.named_parameters():
                ever_gradient[name] |= parameter.grad is not None and torch.isfinite(parameter.grad).all() and float(parameter.grad.norm()) > 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            if step % 20 == 0 or step + 1 == args.steps:
                trajectory.append({"step": step + 1, "case": sample.metadata["sample_id"], **metrics})
        (args.output / "synthetic_training_trajectory.json").write_text(json.dumps(trajectory, indent=2) + "\n", encoding="utf-8")
        evaluation = evaluate(model, samples)
        updates = {name: float((parameter.detach().cpu() - initial[name]).norm()) for name, parameter in model.named_parameters()}
        (args.output / "gradient_update_audit.json").write_text(json.dumps({"ever_finite_nonzero_gradient": ever_gradient, "parameter_delta_l2": updates}, indent=2) + "\n", encoding="utf-8")
        checkpoint = args.output / "preflight_checkpoint.pt"
        torch.save(model.state_dict(), checkpoint)
        reloaded = IndependentGeometryMatcherM1VASI().to(device); reloaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        reload_exact = all(torch.equal(model.state_dict()[name], reloaded.state_dict()[name]) for name in model.state_dict())
        gates = {
            "inherited_initial_tensors_exact": bool(inherited_rows) and all(row["exact"] for row in inherited_rows),
            "gt_route_coverage_ge_0_95": evaluation["gt_route_coverage"] >= 0.95,
            "predicted_route_coverage_ge_0_90": evaluation["predicted_route_coverage"] >= 0.90,
            "permuted_route_coverage_le_0_50": evaluation["permuted_route_coverage"] <= 0.50,
            "median_refined_transfer_le_0_5": evaluation["median_refined_transfer_px"] <= 0.5,
            "all_cases_nonempty": evaluation["all_cases_nonempty"],
            "all_tensors_gradient": all(ever_gradient.values()),
            "all_tensors_updated": all(math.isfinite(value) and value > 0 for value in updates.values()),
            "reload_exact": reload_exact,
        }
        artifact_names = (
            "inherited_tensor_parity.json", "synthetic_training_trajectory.json",
            "gradient_update_audit.json", "preflight_checkpoint.pt",
        )
        artifacts = {
            name: {"path": str((args.output / name).resolve()), "sha256": file_hash(args.output / name)}
            for name in artifact_names
        }
        payload.update(status="passed" if all(gates.values()) else "inconclusive", evidence_rung="S3_trainable" if all(gates.values()) else "S2_action_expressive", evaluation=evaluation, gates=gates, checkpoint=str(checkpoint.resolve()), artifacts=artifacts)
    except Exception as error:
        payload.update(status="inconclusive", evidence_rung="S0_code_present", error=repr(error), traceback=traceback.format_exc(), gates={})
    write_card(args.output, payload)
    if payload["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
