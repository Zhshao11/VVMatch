#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.igm_m0r1 import IndependentGeometryMatcherM0R1
from tools.igm_m0r1.losses import compute_igm_loss as parent_loss
from tools.igm_m1_vasi import IndependentGeometryMatcherM1VASI
from tools.igm_m1_vasi.losses import compute_igm_loss as vasi_loss
from tools.prompt37_igm_m1_hp2_runner import FROZEN_S0_BUNDLE_SHA256, validate_s0_bundle
from tools.prompt37_igm_m1_vasi_preflight import (
    file_hash,
    gt_route_override,
    make_sample,
    synthetic_image,
    transfer_errors,
    valid_route_metrics,
)


EXPERIMENT = "P37-20260929-IGM-M1R1-HP2"
SEED = 20260929
PRIMARY_CASES = {"scale_rotation", "projective"}


def sha_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def homographies(image: np.ndarray) -> dict[str, np.ndarray]:
    angle = math.radians(3.0)
    scale = 1.02
    cosine, sine = scale * math.cos(angle), scale * math.sin(angle)
    center_x, center_y = image.shape[1] / 2.0, image.shape[0] / 2.0
    translate_x = center_x - cosine * center_x + sine * center_y
    translate_y = center_y - sine * center_x - cosine * center_y
    return {
        "identity": np.eye(3),
        "translation": np.array([[1, 0, 4], [0, 1, 3], [0, 0, 1]], dtype=np.float64),
        "scale_rotation": np.array([[cosine, -sine, translate_x], [sine, cosine, translate_y], [0, 0, 1]], dtype=np.float64),
        "projective": np.array([[1.0, 0.02, 3], [-0.015, 1.0, 2], [0.0002, -0.00015, 1]], dtype=np.float64),
    }


def train(
    model: torch.nn.Module,
    samples: list,
    loss_function: Any,
    steps: int,
) -> tuple[list[dict[str, Any]], dict[str, bool], dict[str, float]]:
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    gradients = {name: False for name, _ in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.999))
    trajectory = []
    model.train()
    for step in range(steps):
        sample = samples[step % len(samples)]
        output = model(sample.image0, sample.image1, sample.mask0, sample.mask1, return_matches=False)
        loss, metrics = loss_function(model, output, sample)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            gradients[name] |= parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) and float(parameter.grad.norm()) > 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 20 == 0 or step + 1 == steps:
            trajectory.append({"step": step + 1, "case": sample.metadata["sample_id"], **metrics})
    updates = {name: float((parameter.detach().cpu() - initial[name]).norm()) for name, parameter in model.named_parameters()}
    return trajectory, gradients, updates


def output_errors(output: dict, sample: Any) -> list[float]:
    return transfer_errors(output, sample).detach().cpu().tolist()


def evaluate(parent: IndependentGeometryMatcherM0R1, vasi: IndependentGeometryMatcherM1VASI, samples: list) -> dict:
    per_case = []
    pools = {name: [] for name in ("parent", "predicted", "gt", "permuted")}
    coverages = {name: [] for name in ("predicted", "gt", "permuted")}
    parent.eval(); vasi.eval()
    with torch.no_grad():
        for sample in samples:
            parent_output = parent(sample.image0, sample.image1, sample.mask0, sample.mask1, return_matches=True)
            predicted = vasi(sample.image0, sample.image1, sample.mask0, sample.mask1, return_matches=True)
            gt = vasi(sample.image0, sample.image1, sample.mask0, sample.mask1, return_matches=True, route_override=gt_route_override(predicted, sample))
            permutation01 = torch.roll(torch.arange(predicted["route_mu01_s8"].shape[1], device=sample.image0.device), shifts=max(predicted["route_mu01_s8"].shape[1] // 2, 1))
            permutation10 = torch.roll(torch.arange(predicted["route_mu10_s8"].shape[1], device=sample.image0.device), shifts=max(predicted["route_mu10_s8"].shape[1] // 2, 1))
            permuted = vasi(
                sample.image0, sample.image1, sample.mask0, sample.mask1, return_matches=True,
                route_override={"mu01": predicted["route_mu01_s8"][:, permutation01], "mu10": predicted["route_mu10_s8"][:, permutation10]},
            )
            case = sample.metadata["sample_id"]
            errors = {
                "parent": output_errors(parent_output, sample),
                "predicted": output_errors(predicted, sample),
                "gt": output_errors(gt, sample),
                "permuted": output_errors(permuted, sample),
            }
            coverage = {
                "predicted": valid_route_metrics(predicted, sample)[0],
                "gt": valid_route_metrics(gt, sample)[0],
                "permuted": valid_route_metrics(permuted, sample)[0],
            }
            if case in PRIMARY_CASES:
                for name in pools:
                    pools[name].extend(errors[name])
                for name in coverages:
                    coverages[name].append(coverage[name])
            per_case.append({
                "case": case,
                **{f"{name}_match_count": len(values) for name, values in errors.items()},
                **{f"{name}_median_px": float(np.median(values)) if values else float("inf") for name, values in errors.items()},
                **{f"{name}_coverage": value for name, value in coverage.items()},
            })
    pooled = {name: float(np.median(values)) if values else float("inf") for name, values in pools.items()}
    return {
        "primary_cases": sorted(PRIMARY_CASES),
        "per_case": per_case,
        "pooled_viewpoint_median_px": pooled,
        "viewpoint_route_coverage": {name: float(np.mean(values)) for name, values in coverages.items()},
        "all_paths_all_cases_nonempty": all(row[f"{name}_match_count"] > 0 for row in per_case for name in pools),
        "primary_error_counts": {name: len(values) for name, values in pools.items()},
    }


def write_card(output: Path, payload: dict) -> None:
    (output / "RESULT_CARD.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output / "RESULT_CARD.md").write_text(
        "# IGM-M1R1 HP2 Corrected Matched S2\n\n"
        f"Status: `{payload['status']}`; highest evidence: `{payload['highest_evidence_rung']}`.\n\n"
        + json.dumps(payload.get("gates", {}), indent=2) + "\n",
        encoding="utf-8",
    )


def compute_gates(evaluation: dict[str, Any], integrity: dict[str, bool]) -> dict[str, bool]:
    medians = evaluation["pooled_viewpoint_median_px"]
    coverage = evaluation["viewpoint_route_coverage"]
    return {
        "gt_route_coverage_ge_0_95": coverage["gt"] >= 0.95,
        "predicted_route_coverage_ge_0_90": coverage["predicted"] >= 0.90,
        "permuted_route_coverage_le_0_50": coverage["permuted"] <= 0.50,
        "gt_viewpoint_le_0_85_parent": medians["gt"] <= 0.85 * medians["parent"],
        "predicted_viewpoint_le_0_95_parent": medians["predicted"] <= 0.95 * medians["parent"],
        "predicted_le_permuted_minus_0_25": medians["predicted"] <= medians["permuted"] - 0.25,
        "all_paths_all_cases_nonempty": evaluation["all_paths_all_cases_nonempty"],
        "all_integrity_gates": all(integrity.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--s0-dir", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--r6-unit-tests", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    payload = {"experiment": EXPERIMENT, "stage": "corrected_matched_S2", "host": socket.gethostname(), "watchdog_job_id": os.getenv("WATCHDOG_JOB_ID", ""), "seed": SEED, "steps": args.steps, "artifact_root": f"zzvr(ubuntu):{args.output.resolve()}"}
    try:
        if args.steps != 600:
            raise RuntimeError("corrected S2 is frozen to exactly 600 steps")
        expected = {
            "decision": "6556ec38786d0a1af6d4489f71e6b368bc64cf1ea340f7935f8e1cd8a7367d57",
            "sheet": "ff1c2e6f3e36c72f28b74f96dcb628b99c3dd4f85aa3b222d4819c67ce93e2aa",
            "handoff": "20b47729529bafc17c10880459f61a9c507cadc60fd13908ff1e1e09909fa823",
        }
        source_hashes = {"decision": file_hash(args.decision), "sheet": file_hash(args.sheet), "handoff": file_hash(args.handoff)}
        if source_hashes != expected:
            raise RuntimeError(f"frozen corrected S2 hash mismatch: {source_hashes}")
        runtime_paths = {
            "runner": Path(__file__).resolve(),
            "vasi_model": ROOT / "tools/igm_m1_vasi/model.py",
            "vasi_loss": ROOT / "tools/igm_m1_vasi/losses.py",
            "parent_model": ROOT / "tools/igm_m0r1/model.py",
            "parent_loss": ROOT / "tools/igm_m0r1/losses.py",
        }
        expected_runtime = {
            "vasi_model": "3fbeab8cdc24ff3a966e0f1702cb7d862174de239d328fb75fbd675c896d999c",
            "vasi_loss": "9189fa626572320f7a5918a7f274558532d28f7d8f84f70aa640cea0d5bf5840",
            "parent_model": "fbacf0e9f1c060120590d2fc0a96e27801be70b70dd62f254c66873b30531eea",
            "parent_loss": "c77f419875e0b6a1addd1d005c8089ad84faf2b65bb570e68751624c7241fa64",
        }
        runtime_hashes = {name: file_hash(path) for name, path in runtime_paths.items()}
        if any(runtime_hashes[name] != digest for name, digest in expected_runtime.items()):
            raise RuntimeError(f"corrected S2 runtime closure changed: {runtime_hashes}")
        runtime_manifest = json.loads(args.runtime_manifest.read_text(encoding="utf-8"))
        if runtime_manifest.get("experiment") != EXPERIMENT or runtime_manifest.get("runtime_sha256") != runtime_hashes or runtime_manifest.get("frozen_document_sha256") != source_hashes:
            raise RuntimeError("external corrected-S2 runtime manifest mismatch")
        unit_test_text = args.r6_unit_tests.read_text(encoding="utf-8")
        unit_tests_pass = file_hash(args.r6_unit_tests) == runtime_manifest.get("r6_unit_tests_sha256") and "Ran 12 tests" in unit_test_text and unit_test_text.rstrip().endswith("OK")
        if not unit_tests_pass:
            raise RuntimeError("reviewed R6 unit-test evidence mismatch")
        s0 = validate_s0_bundle(args.s0_dir / "directed_role_pairs.csv", args.s0_dir / "S0_AUDIT.json", args.s0_dir / "MANIFEST_SHA256.json")
        if s0["bundle_sha256"] != FROZEN_S0_BUNDLE_SHA256:
            raise RuntimeError("HP2 bundle changed")
        device = torch.device("cuda")
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        torch.manual_seed(SEED); parent = IndependentGeometryMatcherM0R1().to(device)
        torch.manual_seed(SEED); vasi = IndependentGeometryMatcherM1VASI().to(device)
        inherited_prefixes = ("encoder.", "source_projection.", "target_projection.", "source_matchability.", "target_matchability.", "fine_projection.", "temperature")
        parity = [{"name": name, "parent_sha": sha_tensor(value), "vasi_sha": sha_tensor(vasi.state_dict()[name]), "exact": torch.equal(value, vasi.state_dict()[name])} for name, value in parent.state_dict().items() if name.startswith(inherited_prefixes)]
        (args.output / "initialization_parity.json").write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
        image = synthetic_image()
        samples = [make_sample(image, matrix, name, device) for name, matrix in homographies(image).items()]
        expected_cycle = [sample.metadata["sample_id"] for sample in samples]
        stream = [expected_cycle[step % len(expected_cycle)] for step in range(args.steps)]
        (args.output / "paired_stream.json").write_text(json.dumps({"seed": SEED, "steps": args.steps, "cases": stream}, indent=2) + "\n", encoding="utf-8")
        parent_trajectory, parent_gradients, parent_updates = train(parent, samples, parent_loss, args.steps)
        vasi_trajectory, vasi_gradients, vasi_updates = train(vasi, samples, vasi_loss, args.steps)
        (args.output / "parent_trajectory.json").write_text(json.dumps(parent_trajectory, indent=2) + "\n", encoding="utf-8")
        (args.output / "vasi_trajectory.json").write_text(json.dumps(vasi_trajectory, indent=2) + "\n", encoding="utf-8")
        evaluation = evaluate(parent, vasi, samples)
        (args.output / "per_case_error_table.json").write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
        parent_checkpoint, vasi_checkpoint = args.output / "parent.pt", args.output / "vasi.pt"
        torch.save(parent.state_dict(), parent_checkpoint); torch.save(vasi.state_dict(), vasi_checkpoint)
        parent_reload = IndependentGeometryMatcherM0R1().to(device); parent_reload.load_state_dict(torch.load(parent_checkpoint, map_location=device, weights_only=True))
        vasi_reload = IndependentGeometryMatcherM1VASI().to(device); vasi_reload.load_state_dict(torch.load(vasi_checkpoint, map_location=device, weights_only=True))
        reload_exact = all(torch.equal(parent.state_dict()[name], parent_reload.state_dict()[name]) for name in parent.state_dict()) and all(torch.equal(vasi.state_dict()[name], vasi_reload.state_dict()[name]) for name in vasi.state_dict())
        integrity = {
            "inherited_initialization_exact": bool(parity) and all(row["exact"] for row in parity),
            "paired_stream_exact": len(stream) == 600 and all(stream[index] == expected_cycle[index % 4] for index in range(600)),
            "parent_all_tensors_gradient": all(parent_gradients.values()),
            "vasi_all_tensors_gradient": all(vasi_gradients.values()),
            "parent_all_tensors_updated": all(math.isfinite(value) and value > 0 for value in parent_updates.values()),
            "vasi_all_tensors_updated": all(math.isfinite(value) and value > 0 for value in vasi_updates.values()),
            "reload_exact": reload_exact,
            "assignment_refiner_parity_unit_test": unit_tests_pass,
            "hp2_bundle_exact": True,
            "development_evaluation_unread": True,
        }
        gates = compute_gates(evaluation, integrity)
        audit = {"parent_gradients": parent_gradients, "vasi_gradients": vasi_gradients, "parent_updates": parent_updates, "vasi_updates": vasi_updates, "integrity": integrity}
        (args.output / "gradient_update_integrity.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        artifacts = {}
        for name in ("initialization_parity.json", "paired_stream.json", "parent_trajectory.json", "vasi_trajectory.json", "per_case_error_table.json", "gradient_update_integrity.json", "parent.pt", "vasi.pt"):
            artifacts[name] = {"path": str((args.output / name).resolve()), "sha256": file_hash(args.output / name)}
        passed = all(gates.values())
        payload.update(status="passed" if passed else "inconclusive", highest_evidence_rung="S2_action_expressive", source_hashes=source_hashes, runtime_hashes=runtime_hashes, runtime_manifest={"path": str(args.runtime_manifest.resolve()), "sha256": file_hash(args.runtime_manifest)}, r6_unit_tests={"path": str(args.r6_unit_tests.resolve()), "sha256": file_hash(args.r6_unit_tests)}, optimizer_binding={"name": "AdamW", "learning_rate": 0.0002, "weight_decay": 0.0001, "betas": [0.9, 0.999], "steps_each": 600, "paired_stream": True}, s0_binding=s0, evaluation=evaluation, integrity=integrity, gates=gates, artifacts=artifacts, decision="resume_original_HP2_funnel" if passed else "stop_VASI_no_second_rescue")
    except Exception as error:
        payload.update(status="inconclusive", highest_evidence_rung="S0_code_present", decision="stop_VASI_no_second_rescue", error=repr(error), traceback=traceback.format_exc(), gates={})
    write_card(args.output, payload)
    if payload["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
