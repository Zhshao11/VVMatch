#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.igm_m0r1 import IndependentGeometryMatcherM0R1
from tools.igm_m3_gcrp import GCRPPretrainer
from tools.igm_m3_gcrp.data import load_gcrp_pair
from tools.igm_m3_gcrp.losses import centers, compute_gcrp_loss, contiguous_mask
from tools.prompt37_igm_hp1_common import deterministic_pair, load_pair, move_sample, warp_points


EXPERIMENT = "P37-20260930-IGM-M3-HP3"
SEED = 20260930
HP3_BUNDLE = "8aa746ed71924ce4fa2d15975c2596d218209c73a3f85b7c98a1f071fb8655e1"
ARCHIVE_SHA = "99cf7e1ca167896eb4ca2fe3d903beff6566243a3c0c8e07a13a2596648ad670"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path, role: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return [row for row in csv.DictReader(source) if row["pretrain_role"] == role]


def validate_s0(s0_hashes: Path, pretrain_pairs: Path) -> None:
    if sha(s0_hashes) != HP3_BUNDLE or pretrain_pairs.resolve() != (s0_hashes.parent / "pretrain_directed_pairs.csv").resolve():
        raise RuntimeError("HP3 S0 binding mismatch")
    recorded = json.loads(s0_hashes.read_text())
    for name, digest in recorded.items():
        if sha(s0_hashes.parent / name) != digest:
            raise RuntimeError(f"HP3 S0 artifact changed: {name}")


def validate_runtime_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("experiment") != EXPERIMENT:
        raise RuntimeError("runtime manifest experiment mismatch")
    for record in manifest.get("files", {}).values():
        target = Path(record["path"]).resolve()
        if not target.is_file() or sha(target) != record["sha256"]:
            raise RuntimeError(f"runtime source changed: {target}")
    return {"path": str(path.resolve()), "sha256": sha(path)}


def append(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(row))
        if not exists: writer.writeheader()
        writer.writerow(row)


def scale_masks(output: dict[str, Any], sample: Any) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    output0, output1 = {}, {}
    for name, stride in (("fine", 4), ("coarse", 8), ("guide", 16)):
        feature0, feature1 = output["student_features0"][name], output["student_features1"][name]
        output0[name] = IndependentGeometryMatcherM0R1.coarse_center_mask(sample.mask0, *feature0.shape[-2:], stride=stride)
        output1[name] = IndependentGeometryMatcherM0R1.coarse_center_mask(sample.mask1, *feature1.shape[-2:], stride=stride)
    return output0, output1


def write_card(output: Path, payload: dict) -> None:
    (output / "RESULT_CARD.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output / "RESULT_CARD.md").write_text(f"# IGM-M3 HP3 {payload['stage']} Result\n\nStatus: `{payload['status']}`.\n")


def train(args: argparse.Namespace) -> None:
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    payload = {"experiment": EXPERIMENT, "stage": "P0_pretraining", "arm": args.arm, "host": socket.gethostname(), "watchdog_job_id": os.getenv("WATCHDOG_JOB_ID", ""), "artifact_root": f"zzvr(ubuntu):{args.output.resolve()}"}
    trajectory = args.output / "complete_pretrain_trajectory.csv"
    try:
        if args.steps != 10000 or args.seed != SEED: raise RuntimeError("P0 is frozen to seed 20260930 and 10000 steps")
        validate_s0(args.s0_hashes, args.pretrain_pairs)
        runtime_binding = validate_runtime_manifest(args.runtime_manifest)
        if sha(args.data_archive) != ARCHIVE_SHA: raise RuntimeError("HPatches archive changed")
        fit = rows(args.pretrain_pairs, "fit")
        if len(fit) != 640: raise RuntimeError("fit manifest changed")
        device = torch.device("cuda")
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        model = GCRPPretrainer().to(device)
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        initial = {name: parameter.detach().cpu().clone() for name, parameter in trainable}
        gradients = {name: False for name, _ in trainable}
        ema_initial = {name: value.detach().cpu().clone() for name, value in model.ema_encoder.state_dict().items()}
        optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.999))
        initial_mask, final_mask, times = [], [], []
        stream = (args.output / "exposure_stream.jsonl").open("w")
        for step in range(args.steps):
            started = time.perf_counter(); optimizer.zero_grad(set_to_none=True); metrics_acc = []
            samples = []
            for micro in range(4):
                exposure = step * 4 + micro
                row = deterministic_pair(fit, SEED, exposure)
                sample = move_sample(load_gcrp_pair(args.data_root, row, SEED, exposure), device)
                samples.append(sample); stream.write(json.dumps(sample.metadata, sort_keys=True, default=str) + "\n")
            geometry = torch.cat([sample.homography for sample in samples], 0)
            for micro, sample in enumerate(samples):
                height, width = sample.image0.shape[-2] // 8, sample.image0.shape[-1] // 8
                valid_coarse0 = IndependentGeometryMatcherM0R1.coarse_center_mask(sample.mask0, height, width, stride=8)
                valid_coarse1 = IndependentGeometryMatcherM0R1.coarse_center_mask(sample.mask1, height, width, stride=8)
                mask_generator0 = torch.Generator(device=device).manual_seed(SEED + exposure * 2)
                mask_generator1 = torch.Generator(device=device).manual_seed(SEED + exposure * 2 + 1)
                token_mask0 = contiguous_mask(valid_coarse0, 0.60, mask_generator0)
                token_mask1 = contiguous_mask(valid_coarse1, 0.60, mask_generator1)
                output = model(sample.image0, sample.image1, token_mask0, token_mask1)
                valid0, valid1 = scale_masks(output, sample)
                geometry_index = (micro + 1) % 4 if args.arm == "shuffled" else micro
                loss, metrics = compute_gcrp_loss(model, output, geometry[geometry_index : geometry_index + 1], valid0, valid1, SEED + step * 4 + micro)
                (loss / 4).backward(); metrics_acc.append(metrics)
            for name, parameter in trainable:
                gradients[name] |= parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) and float(parameter.grad.norm()) > 0
            torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 1.0)
            progress = (step + 1) / args.steps
            lr = 2e-4 * ((step + 1) / 500 if step < 500 else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step + 1 - 500) / 9500)))
            for group in optimizer.param_groups: group["lr"] = lr
            optimizer.step(); model.update_ema(0.996); torch.cuda.synchronize(); times.append(time.perf_counter() - started)
            means = {key: statistics.fmean(item[key] for item in metrics_acc) for key in metrics_acc[0]}
            if step < 20: initial_mask.append(means["loss_mask"])
            if step >= args.steps - 20: final_mask.append(means["loss_mask"])
            append(trajectory, {"step": step + 1, "optimizer_steps": step + 1, "processed_samples": (step + 1) * 4, "lr": lr, **means})
        stream.close()
        checkpoint = args.output / f"{args.arm}_step10000.pt"
        export = args.output / f"{args.arm}_encoder_export.pt"
        torch.save({"experiment": EXPERIMENT, "arm": args.arm, "seed": SEED, "step": 10000, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
        torch.save(model.export_encoder(), export)
        reloaded = GCRPPretrainer().to(device)
        reloaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"], strict=True)
        reload_exact = all(torch.equal(model.state_dict()[name], reloaded.state_dict()[name]) for name in model.state_dict())
        updates = {name: float((parameter.detach().cpu() - initial[name]).norm()) for name, parameter in trainable}
        ema_updates = {name: float((value.detach().cpu() - ema_initial[name]).norm()) for name, value in model.ema_encoder.state_dict().items()}
        summary = {"experiment": EXPERIMENT, "arm": args.arm, "steps": 10000, "masked_loss_ratio": statistics.fmean(final_mask) / statistics.fmean(initial_mask), "all_student_tensors_gradient": all(gradients.values()), "all_student_tensors_updated": all(value > 0 for value in updates.values()), "ema_all_changed": all(value > 0 for value in ema_updates.values()), "ema_in_optimizer": any(id(parameter) in {id(value) for group in optimizer.param_groups for value in group["params"]} for parameter in model.ema_encoder.parameters()), "checkpoint_reload_exact": reload_exact, "branch_counts": model.branch_counts, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha(checkpoint), "export": str(export.resolve()), "export_sha256": sha(export), "stream_sha256": sha(args.output / "exposure_stream.jsonl"), "step_seconds_mean": statistics.fmean(times), "step_seconds_p95": float(np.quantile(times, .95))}
        (args.output / "TRAINING_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.output / "gradient_update_audit.json").write_text(json.dumps({"gradients": gradients, "updates": updates, "ema_updates": ema_updates}, indent=2) + "\n")
        payload.update(status="passed", highest_evidence_rung="S3_trainable", summary=summary, runtime_binding=runtime_binding)
    except Exception as error:
        payload.update(status="inconclusive", highest_evidence_rung="S0_code_present", error=repr(error), traceback=traceback.format_exc())
    write_card(args.output, payload)
    if payload["status"] != "passed": raise SystemExit(2)


def probe_metrics(encoder: Any, data_root: Path, probe_rows: list[dict[str, str]], seed: int, device: torch.device) -> tuple[list[dict], float, float, float]:
    result = []
    for index, row in enumerate(probe_rows):
        sample = move_sample(load_pair(data_root, row, seed, index, augment=False), device)
        with torch.no_grad():
            f0 = encoder(sample.image0)[0]; f1 = encoder(sample.image1)[0]
        d0, d1 = F.normalize(f0.flatten(2).transpose(1, 2), dim=-1), F.normalize(f1.flatten(2).transpose(1, 2), dim=-1)
        sim = d0[0] @ d1[0].T
        source = centers(*f0.shape[-2:], 8, device)
        target = centers(*f1.shape[-2:], 8, device)
        projected = warp_points(source[None], sample.homography)[0]
        gt_dist = torch.cdist(projected, target)
        gt = gt_dist.argmin(-1); pred = sim.argmax(-1)
        valid = torch.isfinite(projected).all(-1) & (projected[:, 0] >= 0) & (projected[:, 0] < sample.valid_hw1[1]) & (projected[:, 1] >= 0) & (projected[:, 1] < sample.valid_hw1[0])
        recall = float((torch.linalg.vector_norm(target[pred] - projected, dim=-1)[valid] <= 8).float().mean())
        positive = sim[torch.arange(sim.shape[0], device=device), gt]
        hard = sim.masked_fill(gt_dist <= 16, -1e9).max(-1).values
        margin = float((positive - hard)[valid].mean())
        std = float(d0[0].std(0, unbiased=False).mean())
        result.append({"sequence": row["sequence"], "subset": row["subset"], "recall8": recall, "cosine_margin": margin, "projected_std": std})
    sequence_rows = []
    for sequence in sorted({row["sequence"] for row in result}):
        selected = [row for row in result if row["sequence"] == sequence]
        sequence_rows.append({"sequence": sequence, "subset": selected[0]["subset"], "recall8": statistics.fmean(row["recall8"] for row in selected), "cosine_margin": statistics.fmean(row["cosine_margin"] for row in selected), "projected_std": statistics.fmean(row["projected_std"] for row in selected)})
    return sequence_rows, statistics.fmean(row["recall8"] for row in sequence_rows), statistics.fmean(row["cosine_margin"] for row in sequence_rows), statistics.fmean(row["projected_std"] for row in sequence_rows)


def probe(args: argparse.Namespace) -> None:
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    payload = {"experiment": EXPERIMENT, "stage": "S2_representation_probe", "host": socket.gethostname(), "artifact_root": f"zzvr(ubuntu):{args.output.resolve()}"}
    try:
        validate_s0(args.s0_hashes, args.pretrain_pairs)
        runtime_binding = validate_runtime_manifest(args.runtime_manifest)
        probe_rows = rows(args.pretrain_pairs, "probe")
        if len(probe_rows) != 80 or len({row["sequence"] for row in probe_rows}) != 8: raise RuntimeError("probe manifest changed")
        correct_summary = json.loads((args.correct_export.parent / "TRAINING_SUMMARY.json").read_text())
        shuffled_summary = json.loads((args.shuffled_export.parent / "TRAINING_SUMMARY.json").read_text())
        for expected_arm, summary, export in (("correct", correct_summary, args.correct_export), ("shuffled", shuffled_summary, args.shuffled_export)):
            if summary.get("experiment") != EXPERIMENT or summary.get("arm") != expected_arm or summary.get("steps") != 10000 or summary.get("export_sha256") != sha(export) or summary.get("checkpoint_reload_exact") is not True or summary.get("ema_in_optimizer") is not False:
                raise RuntimeError(f"unfrozen {expected_arm} P0 endpoint")
        if correct_summary["stream_sha256"] != shuffled_summary["stream_sha256"]:
            raise RuntimeError("correct/shuffled exposure streams differ")
        endpoint_freeze = {"experiment": EXPERIMENT, "stage": "P0_endpoint_freeze", "correct": {"export": str(args.correct_export.resolve()), "sha256": sha(args.correct_export), "checkpoint_sha256": correct_summary["checkpoint_sha256"]}, "shuffled": {"export": str(args.shuffled_export.resolve()), "sha256": sha(args.shuffled_export), "checkpoint_sha256": shuffled_summary["checkpoint_sha256"]}, "stream_sha256": correct_summary["stream_sha256"]}
        (args.output / "P0_ENDPOINT_FREEZE.json").write_text(json.dumps(endpoint_freeze, indent=2) + "\n")
        device = torch.device("cuda")
        torch.manual_seed(SEED); random_encoder = IndependentGeometryMatcherM0R1().encoder.to(device)
        correct_encoder = IndependentGeometryMatcherM0R1().encoder.to(device); correct_encoder.load_state_dict(torch.load(args.correct_export, map_location=device, weights_only=True))
        shuffled_encoder = IndependentGeometryMatcherM0R1().encoder.to(device); shuffled_encoder.load_state_dict(torch.load(args.shuffled_export, map_location=device, weights_only=True))
        tables, summaries = {}, {}
        for name, encoder in (("random", random_encoder), ("correct", correct_encoder), ("shuffled", shuffled_encoder)):
            table, recall, margin, std = probe_metrics(encoder, args.data_root, probe_rows, SEED, device)
            tables[name] = table; summaries[name] = {"recall8": recall, "cosine_margin": margin, "projected_std": std}
        correct_training = correct_summary
        viewpoint_positive = sum(c["recall8"] > s["recall8"] for c, s in zip(tables["correct"], tables["shuffled"]) if c["subset"] == "viewpoint")
        gates = {"masked_loss_ratio_le_0_40": correct_training["masked_loss_ratio"] <= .40, "projected_std_ge_0_05": summaries["correct"]["projected_std"] >= .05, "recall_gain_vs_random_pp_ge_15": (summaries["correct"]["recall8"] - summaries["random"]["recall8"]) * 100 >= 15, "recall_gain_vs_shuffled_pp_ge_10": (summaries["correct"]["recall8"] - summaries["shuffled"]["recall8"]) * 100 >= 10, "margin_gain_vs_random_ge_0_15": summaries["correct"]["cosine_margin"] - summaries["random"]["cosine_margin"] >= .15, "margin_gain_vs_shuffled_ge_0_10": summaries["correct"]["cosine_margin"] - summaries["shuffled"]["cosine_margin"] >= .10, "viewpoint_probe_sequences_4of4": viewpoint_positive == 4, "correct_training_integrity": correct_training["all_student_tensors_gradient"] and correct_training["all_student_tensors_updated"] and correct_training["ema_all_changed"] and correct_training["checkpoint_reload_exact"] and not correct_training["ema_in_optimizer"], "exported_extra_inference_parameters_eq_0": set(torch.load(args.correct_export, weights_only=True)) == set(random_encoder.state_dict())}
        (args.output / "probe_tables.json").write_text(json.dumps(tables, indent=2) + "\n")
        passed = all(gates.values()); payload.update(status="passed" if passed else "rejected", highest_evidence_rung="S3_pretraining_capacity", summaries=summaries, gates=gates, runtime_binding=runtime_binding, decision="admit_matcher_tiny" if passed else "stop_exact_GCRP_before_tiny")
    except Exception as error:
        payload.update(status="inconclusive", highest_evidence_rung="S0_code_present", error=repr(error), traceback=traceback.format_exc(), gates={})
    write_card(args.output, payload)
    if payload["status"] != "passed": raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(); sub = root.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train"); training.add_argument("--arm", choices=("correct", "shuffled"), required=True); training.add_argument("--steps", type=int, required=True); training.add_argument("--seed", type=int, required=True); training.add_argument("--data-root", type=Path, required=True); training.add_argument("--data-archive", type=Path, required=True); training.add_argument("--pretrain-pairs", type=Path, required=True); training.add_argument("--s0-hashes", type=Path, required=True); training.add_argument("--runtime-manifest", type=Path, required=True); training.add_argument("--output", type=Path, required=True); training.set_defaults(func=train)
    probing = sub.add_parser("probe"); probing.add_argument("--data-root", type=Path, required=True); probing.add_argument("--pretrain-pairs", type=Path, required=True); probing.add_argument("--s0-hashes", type=Path, required=True); probing.add_argument("--runtime-manifest", type=Path, required=True); probing.add_argument("--correct-export", type=Path, required=True); probing.add_argument("--shuffled-export", type=Path, required=True); probing.add_argument("--output", type=Path, required=True); probing.set_defaults(func=probe)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args(); arguments.func(arguments)
