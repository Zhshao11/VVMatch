#!/usr/bin/env python3
"""Training, preflight, and endpoint evaluation for the frozen IGM HP1 study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.prompt37_igm_hp1_common import (
    PairSample,
    deterministic_pair,
    load_pair,
    metadata_json,
    move_sample,
    read_pair_manifest,
    restore_points,
)


EXPERIMENT = "P37-20260924-IGM-M0R1-HP1"
ARM_NAMES = {"igm_r1": "IGM-M0R1", "official": "OfficialSLiM-HP-Scratch"}
FORBIDDEN_PATH_TOKENS = ("heldout1200", "gate2", "protected", "osdataset", "sar")
FROZEN_S0_BUNDLE_SHA256 = "9718b1a04da78d061ecee010477f4f16e6d7a3c570caa0d7b61075e745149952"
FROZEN_HPATCHES_ARCHIVE_SHA256 = "99cf7e1ca167896eb4ca2fe3d903beff6566243a3c0c8e07a13a2596648ad670"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed64(*parts: object) -> int:
    text_value = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text_value.encode("utf-8")).digest()[:8], "big")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def reject_forbidden_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        lowered = str(path.resolve()).lower()
        if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
            raise RuntimeError(f"forbidden path: {path}")


def validate_s0_bundle(pair_manifest: Path, s0_audit: Path, s0_hashes: Path) -> dict[str, Any]:
    if sha256_file(s0_hashes) != FROZEN_S0_BUNDLE_SHA256:
        raise RuntimeError("S0 bundle digest differs from the frozen inherited digest")
    recorded = json.loads(s0_hashes.read_text(encoding="utf-8"))
    required = {"sequence_roles.csv", "file_manifest.csv", "native_pairs.csv", "directed_role_pairs.csv", "S0_AUDIT.json"}
    if not required.issubset(recorded):
        raise RuntimeError(f"S0 hash manifest missing {sorted(required - set(recorded))}")
    directory = s0_hashes.parent.resolve()
    for name in required:
        path = directory / name
        if not path.is_file() or sha256_file(path) != recorded[name]:
            raise RuntimeError(f"S0 binding mismatch: {path}")
    if pair_manifest.resolve() != (directory / "directed_role_pairs.csv").resolve():
        raise RuntimeError("runner pair manifest is not the hash-bound S0 directed manifest")
    if s0_audit.resolve() != (directory / "S0_AUDIT.json").resolve():
        raise RuntimeError("runner S0 audit is not the hash-bound audit")
    audit = json.loads(s0_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "S0_PASSED" or not all(audit.get("checks", {}).values()):
        raise RuntimeError("S0 audit gate is not fully passed")
    with (directory / "sequence_roles.csv").open(newline="", encoding="utf-8") as handle:
        sequences = list(csv.DictReader(handle))
    with (directory / "file_manifest.csv").open(newline="", encoding="utf-8") as handle:
        files = list(csv.DictReader(handle))
    with pair_manifest.open(newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))
    role_sequences = {role: {row["sequence"] for row in sequences if row["role"] == role} for role in ("train", "development", "evaluation")}
    role_files = {role: {row["relative_path"] for row in files if row["role"] == role} for role in role_sequences}
    if [len(role_sequences[role]) for role in role_sequences] != [72, 18, 18]:
        raise RuntimeError("S0 sequence role counts changed")
    if role_sequences["train"] & role_sequences["development"] or role_sequences["train"] & role_sequences["evaluation"] or role_sequences["development"] & role_sequences["evaluation"]:
        raise RuntimeError("S0 sequence overlap")
    if role_files["train"] & role_files["development"] or role_files["train"] & role_files["evaluation"] or role_files["development"] & role_files["evaluation"]:
        raise RuntimeError("S0 file overlap")
    role_pair_counts = {role: sum(row["role"] == role for row in pairs) for role in role_sequences}
    if role_pair_counts != {"train": 720, "development": 90, "evaluation": 90}:
        raise RuntimeError(f"S0 directed role counts changed: {role_pair_counts}")
    if len({row["sample_id"] for row in pairs}) != len(pairs):
        raise RuntimeError("duplicate directed sample IDs")
    return {"files": {name: recorded[name] for name in sorted(required)}, "bundle_sha256": sha256_file(s0_hashes)}


def verify_role_payload_bytes(data_root: Path, file_manifest: Path, role: str, expected_count: int) -> None:
    with file_manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["role"] == role]
    if len(rows) != expected_count:
        raise RuntimeError(f"{role} file manifest count mismatch: {len(rows)}")
    for row in rows:
        path = (data_root / row["relative_path"]).resolve()
        if not path.is_relative_to(data_root.resolve()) or not path.is_file():
            raise RuntimeError(f"{role} payload path invalid: {path}")
        if path.stat().st_size != int(row["size_bytes"]) or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"{role} payload bytes changed: {path}")


def validate_data_binding(data_root: Path, data_archive: Path, file_manifest: Path, role: str) -> dict[str, str]:
    if not data_archive.is_file() or sha256_file(data_archive) != FROZEN_HPATCHES_ARCHIVE_SHA256:
        raise RuntimeError("HPatches archive identity mismatch")
    verify_role_payload_bytes(data_root, file_manifest, role, 72 * 11 if role == "train" else 18 * 11)
    return {"data_root": str(data_root.resolve()), "archive": str(data_archive.resolve()), "archive_sha256": FROZEN_HPATCHES_ARCHIVE_SHA256}


def validate_approved_source_closure(path: Path) -> dict[str, str]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        records[relative] = digest
    required = set(implementation_bindings())
    if set(records) != required:
        raise RuntimeError(f"approved source closure roles differ: {sorted(set(records) ^ required)}")
    current = implementation_bindings()
    if records != current:
        raise RuntimeError("live runtime source differs from approved exact SHA closure")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "files": records}


def validate_tiny_gate(path: Path) -> dict[str, Any]:
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("experiment") != EXPERIMENT or gate.get("stage") != "tiny" or gate.get("passed") is not True:
        raise RuntimeError("tiny gate is missing, stale, or not passed")
    if gate.get("seed") != 20260923 or gate.get("steps") != 1000 or gate.get("arm") != "IGM-M0R1":
        raise RuntimeError("tiny gate metadata mismatch")
    for role, record in gate.get("artifacts", {}).items():
        artifact = Path(record["path"]).resolve()
        if not artifact.is_file() or sha256_file(artifact) != record.get("sha256"):
            raise RuntimeError(f"tiny gate artifact changed: {role}")
    required = {"checkpoint", "summary", "pair_metrics", "trajectory", "sample_stream", "optimizer_membership", "tensor_gradients", "parameter_deltas", "directed_role_pairs", "file_manifest", "s0_audit", "s0_hashes", "data_archive", "approved_source_hashes"}
    if set(gate.get("artifacts", {})) != required:
        raise RuntimeError("tiny gate artifact closure incomplete")
    artifacts = {role: Path(record["path"]).resolve() for role, record in gate["artifacts"].items()}
    run_artifact_roles = {"checkpoint", "summary", "pair_metrics", "trajectory", "sample_stream", "optimizer_membership", "tensor_gradients", "parameter_deltas"}
    if any(artifacts[role].parent != path.resolve().parent for role in run_artifact_roles):
        raise RuntimeError("tiny artifacts do not belong to the same run root as TINY_GATE.json")
    s0_binding = validate_s0_bundle(artifacts["directed_role_pairs"], artifacts["s0_audit"], artifacts["s0_hashes"])
    if artifacts["file_manifest"].resolve() != (artifacts["s0_hashes"].parent / "file_manifest.csv").resolve():
        raise RuntimeError("tiny file manifest is not the frozen S0 file manifest")
    summary = json.loads(artifacts["summary"].read_text(encoding="utf-8"))
    if summary.get("experiment") != EXPERIMENT or summary.get("stage") != "tiny" or summary.get("arm") != "IGM-M0R1" or summary.get("seed") != 20260923 or summary.get("steps") != 1000:
        raise RuntimeError("tiny summary metadata mismatch")
    checkpoint = torch.load(artifacts["checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("experiment") != EXPERIMENT or checkpoint.get("arm") != "IGM-M0R1" or checkpoint.get("seed") != 20260923 or checkpoint.get("step") != 1000:
        raise RuntimeError("tiny checkpoint metadata mismatch")
    if summary.get("checkpoint_sha256") != gate["artifacts"]["checkpoint"]["sha256"]:
        raise RuntimeError("tiny summary checkpoint hash mismatch")
    from tools.igm_m0r1 import IndependentGeometryMatcherM0R1
    frozen_model = IndependentGeometryMatcherM0R1()
    expected_state = frozen_model.state_dict()
    checkpoint_state = checkpoint.get("state_dict")
    if not isinstance(checkpoint_state, dict) or checkpoint_state.keys() != expected_state.keys():
        raise RuntimeError("tiny checkpoint state_dict is not the frozen M0R1 model")
    for name, expected_tensor in expected_state.items():
        actual_tensor = checkpoint_state[name]
        if not isinstance(actual_tensor, torch.Tensor) or actual_tensor.shape != expected_tensor.shape or actual_tensor.dtype != expected_tensor.dtype or not torch.isfinite(actual_tensor).all():
            raise RuntimeError(f"tiny checkpoint tensor invalid: {name}")
    frozen_model.load_state_dict(checkpoint_state, strict=True)
    expected_parameters = {name: parameter.numel() for name, parameter in frozen_model.named_parameters()}
    if checkpoint.get("parameter_count") != sum(expected_parameters.values()):
        raise RuntimeError("tiny checkpoint parameter count mismatch")
    provenance = checkpoint.get("provenance", {})
    required_provenance = {"s0_bundle_sha256", "hpatches_archive_sha256", "data_root", "sample_stream_sha256", "approved_source_manifest_sha256", "tiny_gate_sha256"}
    if set(provenance) != required_provenance or provenance["s0_bundle_sha256"] != FROZEN_S0_BUNDLE_SHA256 or provenance["hpatches_archive_sha256"] != FROZEN_HPATCHES_ARCHIVE_SHA256 or provenance["tiny_gate_sha256"] is not None:
        raise RuntimeError("tiny checkpoint provenance mismatch")
    data_binding = validate_data_binding(Path(provenance["data_root"]), artifacts["data_archive"], artifacts["file_manifest"], "train")
    approved_binding = validate_approved_source_closure(artifacts["approved_source_hashes"])
    if provenance["approved_source_manifest_sha256"] != approved_binding["sha256"]:
        raise RuntimeError("tiny checkpoint source provenance mismatch")
    if summary.get("s0_binding", {}).get("bundle_sha256") != s0_binding["bundle_sha256"] or summary.get("data_binding") != data_binding or summary.get("approved_source_binding") != approved_binding:
        raise RuntimeError("tiny summary S0/data/source linkage mismatch")
    pair_rows = _read_csv(artifacts["pair_metrics"])
    if len(pair_rows) != 8 or len({row["sample_id"] for row in pair_rows}) != 8:
        raise RuntimeError("tiny pair table must contain exactly eight unique pairs")
    pairs_ge_20_inliers = sum(int(row["inliers_at_3px"]) >= 20 for row in pair_rows)
    every_pair_ge_20_matches = all(int(row["match_count"]) >= 20 for row in pair_rows)
    both_sides_all_null_pairs = sum(float(row["source_null_argmax_rate"]) == 1.0 and float(row["target_null_argmax_rate"]) == 1.0 for row in pair_rows)
    trajectory = _read_csv(artifacts["trajectory"])
    if len(trajectory) != 1000:
        raise RuntimeError("tiny trajectory must contain exactly 1000 optimizer rows")
    for index, row in enumerate(trajectory, start=1):
        if int(row["step"]) != index or int(row["optimizer_steps"]) != index or int(row["processed_samples"]) != index * 4 or not math.isfinite(float(row["loss_total"])):
            raise RuntimeError(f"tiny trajectory discontinuity at step {index}")
    sample_lines = artifacts["sample_stream"].read_text(encoding="utf-8").splitlines()
    if len(sample_lines) != 4000:
        raise RuntimeError("tiny sample stream must contain exactly 4000 exposures")
    sample_stream_hash = gate["artifacts"]["sample_stream"]["sha256"]
    if summary.get("sample_stream_sha256") != sample_stream_hash or checkpoint.get("provenance", {}).get("sample_stream_sha256") != sample_stream_hash:
        raise RuntimeError("tiny sample-stream linkage mismatch")
    frozen_train_rows = read_pair_manifest(artifacts["directed_role_pairs"], "train")
    frozen_tiny_rows = tiny_rows(frozen_train_rows)
    expected_tiny_ids = {row["sample_id"] for row in frozen_tiny_rows}
    if [row["sample_id"] for row in pair_rows] != [row["sample_id"] for row in frozen_tiny_rows]:
        raise RuntimeError("tiny pair metrics are not the exact frozen eight pairs in order")
    observed_tiny_ids = set()
    for exposure, raw_line in enumerate(sample_lines):
        metadata = json.loads(raw_line)
        expected_row = deterministic_pair(frozen_tiny_rows, 20260923, exposure)
        for key in ("sample_id", "native_pair_id", "sequence", "subset", "role", "direction", "source_image", "target_image", "homography"):
            if metadata.get(key) != expected_row[key]:
                raise RuntimeError(f"tiny sample stream mismatch at exposure {exposure}: {key}")
        if str(metadata.get("invert_homography")).lower() != expected_row["invert_homography"].lower() or metadata.get("seed") != 20260923 or metadata.get("exposure") != exposure or metadata.get("rng_seed") != stable_seed64("augmentation", 20260923, exposure, expected_row["sample_id"]):
            raise RuntimeError(f"tiny exposure/RNG replay mismatch at exposure {exposure}")
        observed_tiny_ids.add(metadata["sample_id"])
    if observed_tiny_ids != expected_tiny_ids:
        raise RuntimeError("tiny sample stream does not cover the exact frozen eight pairs")
    initial_loss = statistics.fmean(float(row["loss_total"]) for row in trajectory[:20])
    final_loss = statistics.fmean(float(row["loss_total"]) for row in trajectory[-20:])
    recomputed_loss_ratio = final_loss / initial_loss
    if not math.isclose(float(summary["initial_loss_mean"]), initial_loss, rel_tol=0.0, abs_tol=1e-9) or not math.isclose(float(summary["final_loss_mean"]), final_loss, rel_tol=0.0, abs_tol=1e-9) or not math.isclose(float(summary["loss_ratio"]), recomputed_loss_ratio, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("tiny loss summary differs from complete trajectory")
    deltas = _read_csv(artifacts["parameter_deltas"])
    gradients = _read_csv(artifacts["tensor_gradients"])
    membership = json.loads(artifacts["optimizer_membership"].read_text(encoding="utf-8"))
    optimizer_items = [item for values in membership.get("groups", {}).values() for item in values]
    optimizer_names = [item["name"] for item in optimizer_items]
    delta_names = [row["name"] for row in deltas]
    if len(optimizer_names) != len(set(optimizer_names)) or set(optimizer_names) != set(delta_names) or set(optimizer_names) != set(expected_parameters):
        raise RuntimeError("tiny optimizer membership is not an exact parameter partition")
    if any(int(item["numel"]) != expected_parameters[item["name"]] for item in optimizer_items):
        raise RuntimeError("tiny optimizer parameter shape/count mismatch")
    all_updates = bool(deltas) and all(str(row["updated"]).lower() == "true" and math.isfinite(float(row["delta_l2"])) and float(row["delta_l2"]) > 0 for row in deltas)
    gradient_pass_by_name = {name: False for name in delta_names}
    for row in gradients:
        if row["name"] in gradient_pass_by_name:
            value = float(row["gradient_l2"])
            gradient_pass_by_name[row["name"]] |= str(row["finite_nonzero"]).lower() == "true" and math.isfinite(value) and value > 0
    all_gradients = bool(gradient_pass_by_name) and all(gradient_pass_by_name.values())
    recomputed = bool(
        recomputed_loss_ratio <= 0.25
        and pairs_ge_20_inliers >= 7
        and every_pair_ge_20_matches
        and both_sides_all_null_pairs == 0
        and all_gradients
        and all_updates
        and summary.get("checkpoint_reload_exact") is True
    )
    expected_values = {
        "loss_ratio": recomputed_loss_ratio,
        "pairs_ge_20_inliers": pairs_ge_20_inliers,
        "every_pair_ge_20_matches": every_pair_ge_20_matches,
        "both_sides_all_null_pairs": both_sides_all_null_pairs,
        "gradient_tensors": all_gradients,
        "update_tensors": all_updates,
        "reload_exact": summary["checkpoint_reload_exact"],
    }
    if gate.get("gate_values") != expected_values or gate.get("passed") is not recomputed or not recomputed:
        raise RuntimeError("tiny gate does not equal the recomputed frozen conjunction")
    return gate


def validate_conditional_tiny_gate(stage: str, path: Path | None) -> dict[str, Any] | None:
    if stage not in ("preflight", "formal"):
        return None
    if path is None:
        raise RuntimeError("passing frozen tiny gate is mandatory before budget preflight/formal")
    return validate_tiny_gate(path)


def validate_launch_guard(source_manifest_sha256: str) -> None:
    if os.getenv("P37_M0R1_SOURCE_ANCHOR_VERIFIED") != source_manifest_sha256:
        raise RuntimeError("approved source launch digest is missing")
    try:
        guard_pid = int(os.environ["P37_M0R1_GUARD_PID"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("approved launch guard PID is missing") from error
    if guard_pid != os.getppid():
        raise RuntimeError("runner parent is not the approved launch guard process")
    command_line = Path(f"/proc/{guard_pid}/cmdline").read_bytes().split(b"\0")
    expected_guard = (ROOT / "tools/prompt37_igm_m0r1_launch_guard.py").resolve()
    resolved = {Path(value.decode()).resolve() for value in command_line if value and b"prompt37_igm_m0r1_launch_guard.py" in value}
    if expected_guard not in resolved:
        raise RuntimeError("runner parent command is not the approved launch guard path")


def validate_stage_settings(stage: str, seed: int, steps: int) -> None:
    if stage == "smoke" and (seed != 20260923 or steps != 10):
        raise RuntimeError("S1 smoke is fixed to seed 20260923 and ten optimizer steps")
    if stage == "tiny" and (seed != 20260923 or steps != 1000):
        raise RuntimeError("tiny is frozen to seed 20260923 and exactly 1000 optimizer steps")
    if stage == "preflight" and not 10 <= steps <= 100:
        raise RuntimeError("budget preflight requires 10..100 real optimizer steps")
    if stage == "formal" and (seed not in (20260923, 20260924) or steps != 20000):
        raise RuntimeError("formal is frozen to seeds 20260923/20260924 and exactly 20000 steps")


def implementation_bindings() -> dict[str, str]:
    relative = (
        "tools/prompt37_igm_hp1_common.py",
        "tools/prompt37_igm_m0r1_hp1_runner.py",
        "tools/official_slim_hp_scratch_adapter.py",
        "tools/igm_m0r1/model.py",
        "tools/igm_m0r1/losses.py",
    )
    return {name: sha256_file(ROOT / name) for name in relative}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


class IGMAdapter:
    def __init__(self, seed: int, device: torch.device) -> None:
        from tools.igm_m0r1 import IndependentGeometryMatcherM0R1
        from tools.igm_m0r1.losses import compute_igm_loss

        seed_everything(seed)
        self.model = IndependentGeometryMatcherM0R1().to(device)
        self.compute_loss = compute_igm_loss
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

    def training_loss(self, sample: PairSample) -> tuple[torch.Tensor, dict[str, float]]:
        output = self.model(sample.image0, sample.image1, sample.mask0, sample.mask1, return_matches=False)
        return self.compute_loss(self.model, output, sample)

    @torch.no_grad()
    def infer(self, sample: PairSample) -> dict[str, torch.Tensor]:
        output = self.model(sample.image0, sample.image1, sample.mask0, sample.mask1, return_matches=True)
        matches = output["matches"]
        return {
            "source_xy": matches.source_xy,
            "target_xy": matches.target_xy,
            "confidence": matches.confidence,
            "source_null_argmax_rate": (output["source_assignment"].argmax(-1) == output["source_assignment"].shape[-1] - 1).float().mean(),
            "target_null_argmax_rate": (output["target_assignment"].argmax(-1) == output["target_assignment"].shape[-1] - 1).float().mean(),
        }

    def parameter_groups(self) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
        named = dict(self.model.named_parameters())
        prefixes = {
            "encoder_and_fpn": ("encoder.",),
            "dense_pair_interaction": ("interaction.",),
            "affinity_and_matchability": (
                "source_projection.", "target_projection.", "source_matchability.", "target_matchability.",
            ),
            "bilateral_fine_projection": ("fine_projection.",),
        }
        return {
            group: [(name, parameter) for name, parameter in named.items() if name.startswith(group_prefixes)]
            for group, group_prefixes in prefixes.items()
        }


def build_adapter(arm: str, seed: int, device: torch.device, source_root: Path):
    if arm == "igm_r1":
        return IGMAdapter(seed, device)
    if arm == "official":
        from tools.official_slim_hp_scratch_adapter import OfficialScratchAdapter

        return OfficialScratchAdapter(source_root, seed, device)
    raise ValueError(arm)


def parameter_groups(adapter: Any, arm: str) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    if arm == "igm_r1":
        return adapter.parameter_groups()
    named = dict(adapter.model.named_parameters())
    prefixes = {
        "feature_backbone": ("feature_backbone.",),
        "recurrent_refinement": ("recurrent_refine_unit.",),
        "temperatures": ("coarse_temperature", "fine_temperature"),
    }
    return {
        group: [(name, parameter) for name, parameter in named.items() if name.startswith(group_prefixes)]
        for group, group_prefixes in prefixes.items()
    }


def group_norms(groups: dict[str, list[tuple[str, torch.nn.Parameter]]], kind: str) -> dict[str, float]:
    output = {}
    for group, values in groups.items():
        if kind == "gradient":
            tensors = [p.grad.detach().float() for _, p in values if p.grad is not None]
        else:
            tensors = [p.detach().float() for _, p in values]
        output[group] = math.sqrt(sum(float(t.square().sum()) for t in tensors)) if tensors else 0.0
    return output


def tiny_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    sequence_order = []
    for row in rows:
        if row["sequence"] not in sequence_order:
            sequence_order.append(row["sequence"])
    selected = []
    for sequence in sequence_order[:4]:
        forward = [row for row in rows if row["sequence"] == sequence and row["direction"] == "forward"]
        selected.extend(forward[:2])
    if len(selected) != 8:
        raise RuntimeError(f"tiny selection has {len(selected)} rows")
    return selected


def lr_at(step: int, total_steps: int) -> float:
    base = 2e-4
    if step < 500:
        return base * (step + 1) / 500
    progress = (step + 1 - 500) / max(total_steps - 500, 1)
    return base * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def inlier_count(prediction: dict[str, torch.Tensor], sample: PairSample, threshold: float = 3.0) -> int:
    if prediction["source_xy"].numel() == 0:
        return 0
    from tools.prompt37_igm_hp1_common import warp_points

    target = warp_points(prediction["source_xy"][None], sample.homography)[0]
    return int((torch.linalg.vector_norm(target - prediction["target_xy"], dim=1) <= threshold).sum())


def save_checkpoint(path: Path, adapter: Any, arm: str, seed: int, step: int, provenance: dict[str, Any]) -> None:
    torch.save(
        {"experiment": EXPERIMENT, "arm": ARM_NAMES[arm], "seed": seed, "step": step, "parameter_count": sum(parameter.numel() for parameter in adapter.parameters()), "provenance": provenance, "state_dict": adapter.state_dict()},
        path,
    )


def exact_reload_audit(adapter: Any, arm: str, seed: int, device: torch.device, source_root: Path, checkpoint: Path) -> bool:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    reloaded = build_adapter(arm, seed, device, source_root)
    reloaded.load_state_dict(payload["state_dict"])
    current = adapter.state_dict()
    return current.keys() == reloaded.state_dict().keys() and all(
        torch.equal(current[key], reloaded.state_dict()[key]) for key in current
    )


def run_training(args: argparse.Namespace) -> None:
    reject_forbidden_paths((args.data_root, args.data_archive, args.pair_manifest, args.output))
    validate_stage_settings(args.stage, args.seed, args.steps)
    s0_binding = validate_s0_bundle(args.pair_manifest, args.s0_audit, args.s0_hashes)
    data_binding = validate_data_binding(args.data_root, args.data_archive, args.s0_hashes.parent / "file_manifest.csv", "train")
    source_binding = validate_approved_source_closure(args.approved_source_hashes)
    if args.stage in ("tiny", "preflight", "formal"):
        validate_launch_guard(source_binding["sha256"])
    tiny_gate_binding = validate_conditional_tiny_gate(args.stage, args.tiny_gate)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda")
    rows = read_pair_manifest(args.pair_manifest, "train")
    if args.stage == "tiny":
        rows = tiny_rows(rows)
    adapter = build_adapter(args.arm, args.seed, device, args.official_source)
    adapter.train()
    groups = parameter_groups(adapter, args.arm)
    if not groups or any(not values for values in groups.values()):
        raise RuntimeError(f"empty parameter group: {groups.keys()}")
    parameters = list(adapter.parameters())
    named_parameters = dict(adapter.model.named_parameters())
    grouped_ids = [id(parameter) for values in groups.values() for _, parameter in values]
    trainable_ids = [id(parameter) for parameter in parameters if parameter.requires_grad]
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != set(trainable_ids):
        raise RuntimeError("optimizer/group membership is not an exact one-to-one partition")
    optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.999))
    membership = {
        "arm": ARM_NAMES[args.arm],
        "parameter_count": sum(p.numel() for p in parameters),
        "groups": {group: [{"name": name, "id": id(parameter), "numel": parameter.numel()} for name, parameter in values] for group, values in groups.items()},
    }
    write_json(args.output / "optimizer_parameter_IDs.json", membership)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in named_parameters.items()}
    ever_finite_nonzero_gradient = {name: False for name in named_parameters}
    sample_log = (args.output / "sample_stream.jsonl").open("w", encoding="utf-8")
    trajectory_path = args.output / "complete_training_and_action_trajectory.csv"
    grad_path = args.output / "component_group_gradient_table.csv"
    parameter_grad_path = args.output / "component_parameter_gradient_table.csv"
    step_times = []
    initial_losses = []
    final_losses = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(args.steps):
        started = time.perf_counter()
        accumulated_metrics: dict[str, list[float]] = {}
        for micro in range(4):
            exposure = step * 4 + micro
            row = deterministic_pair(rows, args.seed, exposure)
            sample = move_sample(load_pair(args.data_root, row, args.seed, exposure, augment=True), device)
            sample_log.write(metadata_json(sample.metadata) + "\n")
            loss, metrics = adapter.training_loss(sample)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at step {step} micro {micro}")
            (loss / 4.0).backward()
            for key, value in metrics.items():
                accumulated_metrics.setdefault(key, []).append(float(value))
        gradients = group_norms(groups, "gradient")
        if any(not math.isfinite(value) or value <= 0 for value in gradients.values()):
            raise RuntimeError(f"gradient gate failed at step {step}: {gradients}")
        parameter_gradients = {}
        for name, parameter in named_parameters.items():
            norm = float(parameter.grad.detach().float().norm()) if parameter.grad is not None else float("nan")
            parameter_gradients[name] = norm
            ever_finite_nonzero_gradient[name] |= math.isfinite(norm) and norm > 0
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        learning_rate = lr_at(step, args.steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        step_times.append(elapsed)
        means = {key: statistics.fmean(values) for key, values in accumulated_metrics.items()}
        loss_value = means["loss_total"]
        if step < min(20, args.steps):
            initial_losses.append(loss_value)
        if step >= max(0, args.steps - 20):
            final_losses.append(loss_value)
        trajectory = {
            "step": step + 1,
            "optimizer_steps": step + 1,
            "processed_samples": (step + 1) * 4,
            "learning_rate": learning_rate,
            "step_seconds": elapsed,
            **means,
        }
        append_csv(trajectory_path, trajectory)
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            append_csv(grad_path, {"step": step + 1, **gradients})
            for name, norm in parameter_gradients.items():
                append_csv(parameter_grad_path, {"step": step + 1, "name": name, "gradient_l2": norm, "finite_nonzero": math.isfinite(norm) and norm > 0})
    sample_log.close()
    sample_stream_hash = sha256_file(args.output / "sample_stream.jsonl")
    checkpoint = args.output / f"{args.arm}_step_{args.steps:05d}.pt"
    checkpoint_provenance = {
        "s0_bundle_sha256": FROZEN_S0_BUNDLE_SHA256,
        "hpatches_archive_sha256": FROZEN_HPATCHES_ARCHIVE_SHA256,
        "data_root": str(args.data_root.resolve()),
        "sample_stream_sha256": sample_stream_hash,
        "approved_source_manifest_sha256": source_binding["sha256"],
        "tiny_gate_sha256": sha256_file(args.tiny_gate) if args.tiny_gate is not None else None,
    }
    save_checkpoint(checkpoint, adapter, args.arm, args.seed, args.steps, checkpoint_provenance)
    checkpoint_hash = sha256_file(checkpoint)
    reload_exact = exact_reload_audit(adapter, args.arm, args.seed, device, args.official_source, checkpoint)
    delta_rows = []
    parameter_to_group = {name: group for group, values in groups.items() for name, _ in values}
    for name, parameter in named_parameters.items():
        delta = float((parameter.detach().cpu() - initial[name]).float().norm())
        delta_rows.append({"name": name, "group": parameter_to_group[name], "delta_l2": delta, "numel": parameter.numel(), "updated": math.isfinite(delta) and delta > 0})
    write_csv(args.output / "initial_final_parameter_delta_table.csv", delta_rows)
    update_by_group = {group: sum(row["delta_l2"] for row in delta_rows if row["group"] == group) for group in groups}
    gradient_tensor_gate = all(ever_finite_nonzero_gradient.values())
    update_tensor_gate = all(row["updated"] for row in delta_rows)
    tensor_gate_required = args.arm == "igm_r1"
    if tensor_gate_required and not gradient_tensor_gate:
        missing = [name for name, passed in ever_finite_nonzero_gradient.items() if not passed]
        raise RuntimeError(f"per-tensor gradient gate failed: {missing}")
    if tensor_gate_required and not update_tensor_gate:
        missing = [row["name"] for row in delta_rows if not row["updated"]]
        raise RuntimeError(f"per-tensor update gate failed: {missing}")
    tiny_rows_result = []
    if args.stage == "tiny":
        adapter.eval()
        for index, row in enumerate(rows):
            sample = move_sample(load_pair(args.data_root, row, args.seed, index, augment=False), device)
            prediction = adapter.infer(sample)
            tiny_rows_result.append({
                "sample_id": row["sample_id"],
                "inliers_at_3px": inlier_count(prediction, sample),
                "match_count": prediction["source_xy"].shape[0],
                "source_null_argmax_rate": float(prediction.get("source_null_argmax_rate", torch.tensor(float("nan")))),
                "target_null_argmax_rate": float(prediction.get("target_null_argmax_rate", torch.tensor(float("nan")))),
            })
        write_csv(args.output / "tiny_pair_metrics.csv", tiny_rows_result)
    summary = {
        "experiment": EXPERIMENT,
        "stage": args.stage,
        "arm": ARM_NAMES[args.arm],
        "seed": args.seed,
        "host": socket.gethostname(),
        "watchdog_job_id": os.getenv("WATCHDOG_JOB_ID", ""),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "steps": args.steps,
        "effective_batch": 4,
        "precision": "fp32",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_reload_exact": reload_exact,
        "initial_loss_mean": statistics.fmean(initial_losses),
        "final_loss_mean": statistics.fmean(final_losses),
        "loss_ratio": statistics.fmean(final_losses) / statistics.fmean(initial_losses),
        "parameter_count": membership["parameter_count"],
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "step_seconds_mean": statistics.fmean(step_times),
        "step_seconds_p95": float(np.quantile(step_times, 0.95)),
        "gradient_groups_all_finite_nonzero": all(value > 0 and math.isfinite(value) for value in gradients.values()),
        "gradient_tensors_all_finite_nonzero": gradient_tensor_gate,
        "per_tensor_gate_required": tensor_gate_required,
        "update_norm_by_group": update_by_group,
        "update_groups_all_nonzero": all(value > 0 for value in update_by_group.values()),
        "update_tensors_all_nonzero": update_tensor_gate,
        "tiny_pairs": tiny_rows_result,
        "tiny_pairs_ge_20_inliers": sum(row["inliers_at_3px"] >= 20 for row in tiny_rows_result),
        "tiny_every_pair_ge_20_matches": bool(tiny_rows_result) and all(row["match_count"] >= 20 for row in tiny_rows_result),
        "tiny_both_sides_all_null_pairs": sum(row.get("source_null_argmax_rate") == 1.0 and row.get("target_null_argmax_rate") == 1.0 for row in tiny_rows_result),
        "s0_binding": s0_binding,
        "data_binding": data_binding,
        "approved_source_binding": source_binding,
        "sample_stream_sha256": sample_stream_hash,
        "tiny_gate_binding": tiny_gate_binding,
        "implementation_bindings": implementation_bindings(),
        "optimizer_binding": {"name": "AdamW", "learning_rate": 2e-4, "weight_decay": 1e-4, "betas": [0.9, 0.999], "warmup_steps": 500, "final_lr_fraction": 0.05, "global_grad_clip": 1.0},
        "sample_stream_binding": "SHA256 pair-order + per-exposure SHA256 augmentation seed; four sequential microbatches per optimizer step",
    }
    write_json(args.output / "TRAINING_SUMMARY.json", summary)
    if args.stage == "tiny":
        tiny_passed = bool(
            summary["loss_ratio"] <= 0.25
            and summary["tiny_pairs_ge_20_inliers"] >= 7
            and summary["tiny_every_pair_ge_20_matches"]
            and summary["tiny_both_sides_all_null_pairs"] == 0
            and summary["gradient_tensors_all_finite_nonzero"]
            and summary["update_tensors_all_nonzero"]
            and summary["checkpoint_reload_exact"]
        )
        artifact_paths = {
            "checkpoint": checkpoint,
            "summary": args.output / "TRAINING_SUMMARY.json",
            "pair_metrics": args.output / "tiny_pair_metrics.csv",
            "trajectory": trajectory_path,
            "sample_stream": args.output / "sample_stream.jsonl",
            "optimizer_membership": args.output / "optimizer_parameter_IDs.json",
            "tensor_gradients": parameter_grad_path,
            "parameter_deltas": args.output / "initial_final_parameter_delta_table.csv",
            "directed_role_pairs": args.pair_manifest,
            "file_manifest": args.s0_hashes.parent / "file_manifest.csv",
            "s0_audit": args.s0_audit,
            "s0_hashes": args.s0_hashes,
            "data_archive": args.data_archive,
            "approved_source_hashes": args.approved_source_hashes,
        }
        gate = {
            "experiment": EXPERIMENT,
            "stage": "tiny",
            "arm": ARM_NAMES[args.arm],
            "seed": args.seed,
            "steps": args.steps,
            "passed": tiny_passed,
            "gate_values": {
                "loss_ratio": summary["loss_ratio"],
                "pairs_ge_20_inliers": summary["tiny_pairs_ge_20_inliers"],
                "every_pair_ge_20_matches": summary["tiny_every_pair_ge_20_matches"],
                "both_sides_all_null_pairs": summary["tiny_both_sides_all_null_pairs"],
                "gradient_tensors": summary["gradient_tensors_all_finite_nonzero"],
                "update_tensors": summary["update_tensors_all_nonzero"],
                "reload_exact": summary["checkpoint_reload_exact"],
            },
            "artifacts": {role: {"path": str(path.resolve()), "sha256": sha256_file(path)} for role, path in artifact_paths.items()},
        }
        write_json(args.output / "TINY_GATE.json", gate)
    print(json.dumps(summary, indent=2))


def corner_error(h_est: np.ndarray | None, h_gt: np.ndarray, width: int, height: int) -> float:
    if h_est is None or not np.isfinite(h_est).all():
        return float("inf")
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float64).reshape(-1, 1, 2)
    estimated = cv2.perspectiveTransform(corners, h_est).reshape(-1, 2)
    truth = cv2.perspectiveTransform(corners, h_gt).reshape(-1, 2)
    return float(np.linalg.norm(estimated - truth, axis=1).mean() / (min(width, height) / 480.0))


def auc(errors: list[float], threshold: float) -> float:
    values = np.sort(np.asarray(errors, dtype=np.float64))
    recall = (np.arange(len(values)) + 1) / max(len(values), 1)
    values = np.r_[0.0, values]
    recall = np.r_[0.0, recall]
    index = int(np.searchsorted(values, threshold))
    if index <= 0:
        return 0.0
    return float(np.trapz(np.r_[recall[:index], recall[index - 1]], x=np.r_[values[:index], threshold]) / threshold)


REQUIRED_FREEZE_BINDINGS = {
    "method_spec", "function_contract", "budget_and_gates", "experiment_sheet",
    "sequence_roles", "file_manifest", "native_pairs", "directed_role_pairs",
    "s0_audit", "s0_hashes", "evaluator_source", "common_source",
    "igm_model_source", "igm_loss_source", "official_adapter_source",
    "official_source_manifest", "hpatches_archive", "tiny_gate", "approved_source_hashes",
}


def validate_frozen_endpoints_only(freeze: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    if freeze.get("experiment") != EXPERIMENT or freeze.get("decision") != "EVALUATE_FINAL_ENDPOINTS":
        raise RuntimeError("endpoint freeze decision/experiment mismatch")
    endpoints = freeze.get("endpoints", [])
    expected = {(candidate_arm, candidate_seed) for candidate_arm in ARM_NAMES for candidate_seed in (20260923, 20260924)}
    actual = {(entry.get("arm"), entry.get("seed")) for entry in endpoints}
    if len(endpoints) != 4 or actual != expected:
        raise RuntimeError("freeze must contain exactly all four arm/seed endpoints")
    output = {}
    for entry in endpoints:
        if entry.get("step") != 20000:
            raise RuntimeError("non-final endpoint in freeze")
        checkpoint = Path(entry["checkpoint"]).resolve()
        if not checkpoint.is_file() or sha256_file(checkpoint) != entry.get("checkpoint_sha256"):
            raise RuntimeError("endpoint checkpoint hash mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("experiment") != EXPERIMENT or payload.get("arm") != ARM_NAMES[entry["arm"]] or payload.get("seed") != entry["seed"] or payload.get("step") != 20000:
            raise RuntimeError("endpoint checkpoint metadata mismatch")
        if not isinstance(payload.get("parameter_count"), int) or payload["parameter_count"] <= 0:
            raise RuntimeError("endpoint checkpoint parameter count metadata missing")
        provenance = payload.get("provenance", {})
        required_provenance = {"s0_bundle_sha256", "hpatches_archive_sha256", "data_root", "sample_stream_sha256", "approved_source_manifest_sha256", "tiny_gate_sha256"}
        if set(provenance) != required_provenance:
            raise RuntimeError("endpoint checkpoint provenance closure incomplete")
        if provenance["s0_bundle_sha256"] != FROZEN_S0_BUNDLE_SHA256 or provenance["hpatches_archive_sha256"] != FROZEN_HPATCHES_ARCHIVE_SHA256:
            raise RuntimeError("endpoint checkpoint data provenance mismatch")
        output[(entry["arm"], entry["seed"])] = {**entry, "checkpoint_parameter_count": payload["parameter_count"], "checkpoint_provenance": provenance}
    for seed in (20260923, 20260924):
        rescue = output[("igm_r1", seed)]["checkpoint_provenance"]
        baseline = output[("official", seed)]["checkpoint_provenance"]
        for key in ("s0_bundle_sha256", "hpatches_archive_sha256", "data_root", "sample_stream_sha256", "approved_source_manifest_sha256", "tiny_gate_sha256"):
            if rescue[key] != baseline[key]:
                raise RuntimeError(f"paired formal endpoint provenance differs for seed {seed}: {key}")
    return output


def verify_external_source_manifest(manifest_path: Path, source_root: Path) -> None:
    records = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, raw_path = line.split(maxsplit=1)
        path = Path(raw_path).resolve()
        records.append((digest, path))
    expected_paths = {source_root.resolve() / "default_config.py"}
    expected_paths.update((source_root.resolve() / "src").rglob("*.py"))
    expected_paths.update((source_root.resolve() / "configs").rglob("*.py"))
    if not records or any(not path.is_relative_to(source_root.resolve()) for _, path in records):
        raise RuntimeError("official source manifest does not describe the live source root")
    if {path for _, path in records} != {path.resolve() for path in expected_paths}:
        raise RuntimeError("official source manifest is not the complete clean Python source closure")
    for digest, path in records:
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"official source binding changed: {path}")


def verify_evaluation_payload_bytes(data_root: Path, file_manifest: Path) -> None:
    with file_manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["role"] == "evaluation"]
    if len(rows) != 18 * 11:
        raise RuntimeError("evaluation file manifest count mismatch")
    for row in rows:
        path = (data_root / row["relative_path"]).resolve()
        if not path.is_relative_to(data_root.resolve()) or not path.is_file():
            raise RuntimeError(f"evaluation payload path invalid: {path}")
        if path.stat().st_size != int(row["size_bytes"]) or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"evaluation payload bytes changed: {path}")


def validate_endpoint_freeze(
    freeze_path: Path,
    selected_checkpoint: Path,
    arm: str,
    seed: int,
    pair_manifest: Path,
    s0_audit: Path,
    s0_hashes: Path,
    data_root: Path,
    data_archive: Path,
    official_source: Path,
    official_source_manifest: Path,
) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    # Endpoint validity is deliberately completed before any evaluation-role
    # manifest, binding hash, or data payload is opened.
    frozen_endpoints = validate_frozen_endpoints_only(freeze)
    selected = frozen_endpoints.get((arm, seed))
    if selected is None or Path(selected["checkpoint"]).resolve() != selected_checkpoint.resolve():
        raise RuntimeError("selected endpoint is not the frozen arm/seed checkpoint")
    bindings = freeze.get("bindings", {})
    if set(bindings) != REQUIRED_FREEZE_BINDINGS:
        raise RuntimeError(f"endpoint bindings mismatch: {sorted(set(bindings) ^ REQUIRED_FREEZE_BINDINGS)}")
    for role, record in bindings.items():
        path = Path(record["path"]).resolve()
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"endpoint binding changed: {role}")
    tiny_gate = validate_tiny_gate(Path(bindings["tiny_gate"]["path"]))
    approved_sources = validate_approved_source_closure(Path(bindings["approved_source_hashes"]["path"]))
    for endpoint in frozen_endpoints.values():
        provenance = endpoint["checkpoint_provenance"]
        if provenance["tiny_gate_sha256"] != bindings["tiny_gate"]["sha256"]:
            raise RuntimeError("formal endpoint is not bound to the frozen passing tiny gate")
        if provenance["approved_source_manifest_sha256"] != approved_sources["sha256"]:
            raise RuntimeError("formal endpoint is not bound to the approved source closure")
    if Path(bindings["evaluator_source"]["path"]).resolve() != Path(__file__).resolve():
        raise RuntimeError("freeze evaluator is not this evaluator source")
    live_sources = {
        "common_source": ROOT / "tools/prompt37_igm_hp1_common.py",
        "igm_model_source": ROOT / "tools/igm_m0r1/model.py",
        "igm_loss_source": ROOT / "tools/igm_m0r1/losses.py",
        "official_adapter_source": ROOT / "tools/official_slim_hp_scratch_adapter.py",
    }
    for role, live_path in live_sources.items():
        if Path(bindings[role]["path"]).resolve() != live_path.resolve():
            raise RuntimeError(f"freeze {role} is not the live imported module")
    if Path(bindings["directed_role_pairs"]["path"]).resolve() != pair_manifest.resolve():
        raise RuntimeError("freeze pair manifest differs from evaluator input")
    s0_directory = s0_hashes.parent.resolve()
    expected_s0_paths = {
        "sequence_roles": s0_directory / "sequence_roles.csv",
        "file_manifest": s0_directory / "file_manifest.csv",
        "native_pairs": s0_directory / "native_pairs.csv",
        "directed_role_pairs": s0_directory / "directed_role_pairs.csv",
        "s0_audit": s0_directory / "S0_AUDIT.json",
        "s0_hashes": s0_directory / "MANIFEST_SHA256.json",
    }
    for role, expected_path in expected_s0_paths.items():
        if Path(bindings[role]["path"]).resolve() != expected_path.resolve():
            raise RuntimeError(f"freeze {role} does not bind the audited S0 bundle")
    validate_s0_bundle(pair_manifest, s0_audit, s0_hashes)
    if Path(bindings["hpatches_archive"]["path"]).resolve() != data_archive.resolve():
        raise RuntimeError("freeze does not bind the supplied HPatches archive")
    if Path(bindings["official_source_manifest"]["path"]).resolve() != official_source_manifest.resolve():
        raise RuntimeError("freeze does not bind the supplied official source manifest")
    verify_external_source_manifest(official_source_manifest, official_source)
    verify_evaluation_payload_bytes(data_root, s0_hashes.parent / "file_manifest.csv")
    return freeze


def sanitize_prediction(prediction: dict[str, torch.Tensor], sample: PairSample) -> dict[str, torch.Tensor]:
    source = prediction["source_xy"]
    target = prediction["target_xy"]
    confidence = prediction["confidence"]
    valid = torch.isfinite(source).all(dim=1) & torch.isfinite(target).all(dim=1) & torch.isfinite(confidence)
    valid &= (source >= 0).all(dim=1) & (target >= 0).all(dim=1)
    valid &= (source[:, 0] < sample.valid_hw0[1]) & (source[:, 1] < sample.valid_hw0[0])
    valid &= (target[:, 0] < sample.valid_hw1[1]) & (target[:, 1] < sample.valid_hw1[0])
    candidates = torch.where(valid)[0]
    order = candidates[torch.argsort(confidence[candidates], descending=True, stable=True)]
    keep = []
    source_seen: set[tuple[float, float]] = set()
    target_seen: set[tuple[float, float]] = set()
    for item in order.tolist():
        source_key = tuple(float(value) for value in source[item].tolist())
        target_key = tuple(float(value) for value in target[item].tolist())
        if source_key in source_seen or target_key in target_seen:
            continue
        source_seen.add(source_key)
        target_seen.add(target_key)
        keep.append(item)
    selected = torch.tensor(keep, dtype=torch.long, device=source.device)[:1000]
    return {"source_xy": source[selected], "target_xy": target[selected], "confidence": confidence[selected]}


def run_evaluation(args: argparse.Namespace) -> None:
    reject_forbidden_paths((args.data_root, args.pair_manifest, args.output, args.checkpoint, args.endpoint_freeze))
    validate_endpoint_freeze(
        args.endpoint_freeze, args.checkpoint, args.arm, args.seed,
        args.pair_manifest, args.s0_audit, args.s0_hashes,
        args.data_root, args.data_archive, args.official_source, args.official_source_manifest,
    )
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    adapter = build_adapter(args.arm, args.seed, device, args.official_source)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    adapter.load_state_dict(payload["state_dict"])
    adapter.eval()
    rows = read_pair_manifest(args.pair_manifest, "evaluation")
    pair_rows = []
    inference_rows = []
    for pair_index, row in enumerate(rows):
        sample = move_sample(load_pair(args.data_root, row, args.seed, pair_index, augment=False), device)
        torch.cuda.synchronize()
        inference_started = time.perf_counter()
        prediction = adapter.infer(sample)
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started
        prediction = sanitize_prediction(prediction, sample)
        source = restore_points(prediction["source_xy"], sample.original_to_network0).cpu().numpy()
        target = restore_points(prediction["target_xy"], sample.original_to_network1).cpu().numpy()
        inference_rows.append({"pair_id": row["native_pair_id"], "sequence": row["sequence"], "inference_seconds": inference_seconds, "matches_after_stable_dedup_cap": len(source)})
        image = cv2.imread(str(args.data_root / row["source_image"]), cv2.IMREAD_GRAYSCALE)
        for repeat in (66, 67, 68, 69, 70, 71):
            cv2.setRNGSeed(repeat)
            network_per_original = float(
                0.5 * (
                    abs(sample.original_to_network1[0, 0, 0].item())
                    + abs(sample.original_to_network1[0, 1, 1].item())
                )
            )
            original_threshold = 3.0 / max(network_per_original, 1e-8)
            estimated = cv2.findHomography(source, target, cv2.RANSAC, original_threshold, maxIters=10000, confidence=0.99999)[0] if len(source) >= 4 else None
            error = corner_error(estimated, sample.original_homography[0].cpu().numpy(), image.shape[1], image.shape[0])
            pair_rows.append({"pair_id": row["native_pair_id"], "sequence": row["sequence"], "subset": row["subset"], "repeat": repeat, "matches": len(source), "corner_error_480px": error})
    write_csv(args.output / "per_pair_repeat.csv", pair_rows)
    write_csv(args.output / "inference_latency.csv", inference_rows)
    aggregate = []
    per_sequence_rows = []
    for repeat in (66, 67, 68, 69, 70, 71):
        repeat_rows = [row for row in pair_rows if row["repeat"] == repeat]
        per_sequence = {}
        for sequence in sorted({row["sequence"] for row in repeat_rows}):
            errors = [row["corner_error_480px"] for row in repeat_rows if row["sequence"] == sequence]
            per_sequence[sequence] = {threshold: auc(errors, threshold) for threshold in (3.0, 5.0, 10.0)}
            per_sequence_rows.append({"repeat": repeat, "sequence": sequence, "subset": next(row["subset"] for row in repeat_rows if row["sequence"] == sequence), **{f"auc@{int(threshold)}": per_sequence[sequence][threshold] for threshold in (3.0, 5.0, 10.0)}})
        aggregate.append({
            "repeat": repeat,
            **{f"sequence_macro_auc@{int(threshold)}": statistics.fmean(value[threshold] for value in per_sequence.values()) for threshold in (3.0, 5.0, 10.0)},
            **{f"pair_weighted_auc@{int(threshold)}": auc([row["corner_error_480px"] for row in repeat_rows], threshold) for threshold in (3.0, 5.0, 10.0)},
        })
    write_csv(args.output / "aggregate_repeats.csv", aggregate)
    write_csv(args.output / "per_sequence_repeat.csv", per_sequence_rows)
    latencies = [row["inference_seconds"] for row in inference_rows]
    artifact_hashes = {
        name: sha256_file(args.output / name)
        for name in ("per_pair_repeat.csv", "per_sequence_repeat.csv", "aggregate_repeats.csv", "inference_latency.csv")
    }
    write_json(args.output / "EVALUATION_SUMMARY.json", {
        "experiment": EXPERIMENT,
        "arm_key": args.arm,
        "arm": ARM_NAMES[args.arm],
        "seed": args.seed,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "sequence_macro_repeat_mean": {f"auc@{int(threshold)}": statistics.fmean(row[f"sequence_macro_auc@{int(threshold)}"] for row in aggregate) for threshold in (3.0, 5.0, 10.0)},
        "pair_weighted_repeat_mean": {f"auc@{int(threshold)}": statistics.fmean(row[f"pair_weighted_auc@{int(threshold)}"] for row in aggregate) for threshold in (3.0, 5.0, 10.0)},
        "inference_seconds_p50": float(np.quantile(latencies, 0.50)),
        "inference_seconds_p95": float(np.quantile(latencies, 0.95)),
        "parameter_count": sum(parameter.numel() for parameter in adapter.parameters()),
        "artifact_hashes": artifact_hashes,
    })


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_aggregate(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    freeze = json.loads(args.endpoint_freeze.read_text(encoding="utf-8"))
    frozen_endpoints = validate_frozen_endpoints_only(freeze)
    records = {}
    for directory in args.evaluation_dir:
        summary = json.loads((directory / "EVALUATION_SUMMARY.json").read_text(encoding="utf-8"))
        key = (summary["arm_key"], int(summary["seed"]))
        if key in records:
            raise RuntimeError(f"duplicate evaluation endpoint {key}")
        frozen = frozen_endpoints.get(key)
        if frozen is None or summary.get("checkpoint_sha256") != frozen.get("checkpoint_sha256"):
            raise RuntimeError(f"evaluation summary is not bound to frozen endpoint {key}")
        artifact_hashes = summary.get("artifact_hashes")
        expected_artifacts = {"per_pair_repeat.csv", "per_sequence_repeat.csv", "aggregate_repeats.csv", "inference_latency.csv"}
        if set(artifact_hashes or {}) != expected_artifacts:
            raise RuntimeError(f"evaluation result manifest incomplete for {key}")
        for name, digest in artifact_hashes.items():
            if sha256_file(directory / name) != digest:
                raise RuntimeError(f"mutated evaluation artifact {key} {name}")
        records[key] = {
            "summary": summary,
            "pairs": _read_csv(directory / "per_pair_repeat.csv"),
            "sequences": _read_csv(directory / "per_sequence_repeat.csv"),
            "latency": _read_csv(directory / "inference_latency.csv"),
        }
    expected = {(arm, seed) for arm in ARM_NAMES for seed in (20260923, 20260924)}
    if set(records) != expected:
        raise RuntimeError(f"joint aggregation requires four endpoints: {set(records)}")
    pair_keys = None
    sequence_keys = None
    for record in records.values():
        current_pair_keys = [(row["pair_id"], int(row["repeat"])) for row in record["pairs"]]
        current_sequence_keys = [(row["sequence"], int(row["repeat"])) for row in record["sequences"]]
        if pair_keys is None:
            pair_keys, sequence_keys = current_pair_keys, current_sequence_keys
        if current_pair_keys != pair_keys or current_sequence_keys != sequence_keys:
            raise RuntimeError("cross-arm/seed pair or sequence key alignment failed")
    seed_deltas = {}
    sequence_deltas: dict[str, list[float]] = {}
    for seed in (20260923, 20260924):
        igm_rows = records[("igm_r1", seed)]["sequences"]
        official_rows = records[("official", seed)]["sequences"]
        deltas = []
        for igm_row, official_row in zip(igm_rows, official_rows):
            delta = float(igm_row["auc@5"]) - float(official_row["auc@5"])
            deltas.append(delta)
            sequence_deltas.setdefault(igm_row["sequence"], []).append(delta)
        seed_deltas[str(seed)] = statistics.fmean(deltas)
    per_sequence_delta = np.asarray([statistics.fmean(sequence_deltas[sequence]) for sequence in sorted(sequence_deltas)], dtype=np.float64)
    rng = np.random.default_rng(20260923)
    indices = rng.integers(0, len(per_sequence_delta), size=(10000, len(per_sequence_delta)))
    bootstrap = per_sequence_delta[indices].mean(axis=1)
    mean_delta = float(per_sequence_delta.mean())
    ci = [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))]
    p95_by_endpoint = {
        key: float(np.quantile([float(row["inference_seconds"]) for row in record["latency"]], 0.95))
        for key, record in records.items()
    }
    igm_p95 = statistics.fmean(p95_by_endpoint[("igm_r1", seed)] for seed in (20260923, 20260924))
    official_p95 = statistics.fmean(p95_by_endpoint[("official", seed)] for seed in (20260923, 20260924))
    igm_parameters = frozen_endpoints[("igm_r1", 20260923)]["checkpoint_parameter_count"]
    official_parameters = frozen_endpoints[("official", 20260923)]["checkpoint_parameter_count"]
    if igm_parameters != frozen_endpoints[("igm_r1", 20260924)]["checkpoint_parameter_count"] or official_parameters != frozen_endpoints[("official", 20260924)]["checkpoint_parameter_count"]:
        raise RuntimeError("parameter count changed across seeds within an arm")
    result = {
        "experiment": EXPERIMENT,
        "pair_sequence_repeat_keys_exactly_aligned": True,
        "bootstrap": {"unit": "sequence", "samples": 10000, "seed": 20260923, "paired_indices": True},
        "mean_auc5_delta_pp": mean_delta * 100.0,
        "auc5_delta_by_seed_pp": {seed: value * 100.0 for seed, value in seed_deltas.items()},
        "paired_sequence_bootstrap_auc5_delta_ci_pp": [value * 100.0 for value in ci],
        "parameter_count_ratio": igm_parameters / official_parameters,
        "inference_p95_ratio": igm_p95 / official_p95,
        "qualification_gates": {
            "mean_delta_ge_minus_3pp": mean_delta * 100.0 >= -3.0,
            "each_seed_gt_minus_5pp": all(value * 100.0 > -5.0 for value in seed_deltas.values()),
            "parameter_ratio_le_1": igm_parameters / official_parameters <= 1.0,
            "inference_p95_ratio_le_1_5": igm_p95 / official_p95 <= 1.5,
        },
        "strong_positive_gates": {
            "mean_delta_ge_0_5pp": mean_delta * 100.0 >= 0.5,
            "both_seed_directions_positive": all(value > 0 for value in seed_deltas.values()),
            "bootstrap_ci_lower_gt_zero": ci[0] > 0,
        },
        "endpoint_summaries": {f"{arm}:{seed}": record["summary"] for (arm, seed), record in records.items()},
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser()
    sub = output.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--stage", choices=("smoke", "tiny", "preflight", "formal"), required=True)
    train.add_argument("--arm", choices=tuple(ARM_NAMES), required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--steps", type=int, required=True)
    train.add_argument("--data-root", type=Path, required=True)
    train.add_argument("--data-archive", type=Path, required=True)
    train.add_argument("--pair-manifest", type=Path, required=True)
    train.add_argument("--s0-audit", type=Path, required=True)
    train.add_argument("--s0-hashes", type=Path, required=True)
    train.add_argument("--official-source", type=Path, required=True)
    train.add_argument("--approved-source-hashes", type=Path, required=True)
    train.add_argument("--tiny-gate", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.set_defaults(func=run_training)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--arm", choices=tuple(ARM_NAMES), required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--data-root", type=Path, required=True)
    evaluate.add_argument("--pair-manifest", type=Path, required=True)
    evaluate.add_argument("--s0-audit", type=Path, required=True)
    evaluate.add_argument("--s0-hashes", type=Path, required=True)
    evaluate.add_argument("--official-source", type=Path, required=True)
    evaluate.add_argument("--official-source-manifest", type=Path, required=True)
    evaluate.add_argument("--data-archive", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--endpoint-freeze", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(func=run_evaluation)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--evaluation-dir", type=Path, action="append", required=True)
    aggregate.add_argument("--endpoint-freeze", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.set_defaults(func=run_aggregate)
    return output


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
