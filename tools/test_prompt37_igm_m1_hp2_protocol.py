from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from tools.igm_m0r1 import IndependentGeometryMatcherM0R1
from tools.igm_m1_vasi import IndependentGeometryMatcherM1VASI
from tools.prompt37_igm_m1_hp2_runner import (
    ARM_NAMES,
    EXPERIMENT,
    FROZEN_S0_BUNDLE_SHA256,
    sha256_file,
    stable_seed64,
    tiny_rows,
    validate_development_gate,
    validate_frozen_endpoints_only,
    validate_s0_bundle,
    validate_stage_settings,
)
from tools.prompt37_igm_hp1_common import deterministic_pair, read_pair_manifest


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT if (ROOT / "research").is_dir() else ROOT.parent
S0 = PROJECT / "research/results/P37-20260926-IGM-M1-HP2/s0_config_audit"


class HP2ProtocolTests(unittest.TestCase):
    def test_frozen_s0_bundle_and_role_isolation(self) -> None:
        binding = validate_s0_bundle(
            S0 / "directed_role_pairs.csv", S0 / "S0_AUDIT.json", S0 / "MANIFEST_SHA256.json"
        )
        self.assertEqual(binding["bundle_sha256"], FROZEN_S0_BUNDLE_SHA256)
        audit = json.loads((S0 / "S0_AUDIT.json").read_text(encoding="utf-8"))
        self.assertTrue(all(audit["checks"].values()))
        with (S0 / "sequence_roles.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        roles = {role: {row["sequence"] for row in rows if row["role"] == role} for role in ("train", "development", "evaluation")}
        self.assertEqual(tuple(map(len, roles.values())), (72, 18, 18))
        self.assertFalse(roles["train"] & roles["development"])
        self.assertFalse(roles["train"] & roles["evaluation"])
        self.assertFalse(roles["development"] & roles["evaluation"])

    def test_stage_settings_and_tiny_stream(self) -> None:
        validate_stage_settings("smoke", 20260926, 10)
        validate_stage_settings("tiny", 20260926, 1500)
        validate_stage_settings("preflight", 20260926, 10)
        validate_stage_settings("preflight", 20260927, 100)
        validate_stage_settings("formal", 20260926, 20000)
        validate_stage_settings("formal", 20260927, 20000)
        for stage, seed, steps in (("tiny", 20260927, 1500), ("formal", 20260928, 20000), ("formal", 20260926, 19999)):
            with self.assertRaises(RuntimeError):
                validate_stage_settings(stage, seed, steps)
        rows = read_pair_manifest(S0 / "directed_role_pairs.csv", "train")
        selected = tiny_rows(rows)
        self.assertEqual(len(selected), 8)
        self.assertEqual(sum(row["subset"] == "viewpoint" for row in selected), 4)
        first = deterministic_pair(selected, 20260926, 0)
        self.assertEqual(first, deterministic_pair(selected, 20260926, 0))
        self.assertEqual(stable_seed64("augmentation", 20260926, 0, first["sample_id"]), stable_seed64("augmentation", 20260926, 0, first["sample_id"]))

    @staticmethod
    def _checkpoint(path: Path, arm: str, seed: int, stream: str) -> dict:
        if arm == "igm_m1":
            model = IndependentGeometryMatcherM1VASI()
            state, count = model.state_dict(), sum(parameter.numel() for parameter in model.parameters())
        elif arm == "igm_r1":
            model = IndependentGeometryMatcherM0R1()
            state, count = model.state_dict(), sum(parameter.numel() for parameter in model.parameters())
        else:
            state, count = {"dummy": torch.tensor([1.0])}, 1
        provenance = {
            "s0_bundle_sha256": FROZEN_S0_BUNDLE_SHA256,
            "hpatches_archive_sha256": "99cf7e1ca167896eb4ca2fe3d903beff6566243a3c0c8e07a13a2596648ad670",
            "data_root": "/frozen/hp2",
            "sample_stream_sha256": stream,
            "approved_source_manifest_sha256": "a" * 64,
            "route_preflight_sha256": "c" * 64,
            "tiny_gate_sha256": "b" * 64,
        }
        torch.save({"experiment": EXPERIMENT, "arm": ARM_NAMES[arm], "seed": seed, "step": 20000, "parameter_count": count, "provenance": provenance, "state_dict": state}, path)
        return {"arm": arm, "seed": seed, "step": 20000, "checkpoint": str(path), "checkpoint_sha256": sha256_file(path)}

    def test_six_endpoint_freeze_is_exact_and_paired(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            endpoints = []
            for seed in (20260926, 20260927):
                stream = f"stream-{seed}"
                for arm in ARM_NAMES:
                    endpoints.append(self._checkpoint(directory / f"{arm}-{seed}.pt", arm, seed, stream))
            freeze = {"experiment": EXPERIMENT, "decision": "EVALUATE_FINAL_ENDPOINTS", "endpoints": endpoints}
            self.assertEqual(len(validate_frozen_endpoints_only(freeze)), 6)
            with self.assertRaises(RuntimeError):
                validate_frozen_endpoints_only({**freeze, "endpoints": endpoints[:-1]})
            payload = torch.load(Path(endpoints[0]["checkpoint"]), map_location="cpu", weights_only=False)
            payload["provenance"]["sample_stream_sha256"] = "unpaired"
            torch.save(payload, Path(endpoints[0]["checkpoint"]))
            endpoints[0]["checkpoint_sha256"] = sha256_file(Path(endpoints[0]["checkpoint"]))
            with self.assertRaises(RuntimeError):
                validate_frozen_endpoints_only(freeze)

    def test_development_gate_is_hash_bound_and_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            freeze = directory / "freeze.json"; freeze.write_text("{}", encoding="utf-8")
            aggregate = directory / "aggregate.json"
            aggregate.write_text(json.dumps({"comparisons": {"m1_minus_m0r1": {"mean_auc5_delta_pp": 5.1, "strata_auc5_delta_pp": {"viewpoint": 10.1, "illumination": -2.9}, "both_seed_directions_positive": True}}}), encoding="utf-8")
            artifacts = {"endpoint_freeze": {"path": str(freeze.resolve()), "sha256": sha256_file(freeze)}, "joint_aggregate": {"path": str(aggregate.resolve()), "sha256": sha256_file(aggregate)}}
            gate = directory / "gate.json"
            gate_values = {"overall_auc5_delta_pp": 5.1, "viewpoint_auc5_delta_pp": 10.1, "both_seed_directions_positive": True, "illumination_auc5_delta_pp": -2.9}
            gate.write_text(json.dumps({"experiment": EXPERIMENT, "role": "development", "passed": True, "gate_values": gate_values, "artifacts": artifacts}), encoding="utf-8")
            validate_development_gate(gate, freeze)
            payload = json.loads(gate.read_text(encoding="utf-8")); payload["passed"] = False
            gate.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_development_gate(gate, freeze)


if __name__ == "__main__":
    unittest.main(verbosity=2)
