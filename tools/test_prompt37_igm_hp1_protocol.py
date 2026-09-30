from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import torch

from tools.prompt37_igm_hp1_runner import (
    ARM_NAMES,
    EXPERIMENT,
    REQUIRED_FREEZE_BINDINGS,
    sha256_file,
    validate_endpoint_freeze,
    validate_s0_bundle,
    validate_stage_settings,
    run_aggregate,
    verify_evaluation_payload_bytes,
)


class ProtocolGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configured = os.getenv("P37_IGM_TEST_REMOTE_ROOT")
        if not configured:
            raise unittest.SkipTest("P37_IGM_TEST_REMOTE_ROOT is required")
        cls.root = Path(configured)
        cls.source = cls.root / "source"
        cls.s0 = cls.root / "s0_config_audit"
        cls.data_root = cls.root / "data/hpatches-sequences-release"
        cls.data_archive = cls.root / "data/hpatches-sequences-release.zip"
        cls.official_source = Path("/mnt/sda/zhushihao/图像匹配/SLiM_clean_fa0_8b34762")

    def test_stage_and_s0_bindings(self) -> None:
        validate_stage_settings("tiny", 20260923, 1000)
        validate_stage_settings("formal", 20260924, 20000)
        with self.assertRaises(RuntimeError):
            validate_stage_settings("formal", 20260925, 20000)
        with self.assertRaises(RuntimeError):
            validate_stage_settings("formal", 20260923, 19999)
        validate_s0_bundle(
            self.s0 / "directed_role_pairs.csv",
            self.s0 / "S0_AUDIT.json",
            self.s0 / "MANIFEST_SHA256.json",
        )
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "s0"
            shutil.copytree(self.s0, copied)
            with (copied / "directed_role_pairs.csv").open("a", encoding="utf-8") as handle:
                handle.write("mutation")
            with self.assertRaises(RuntimeError):
                validate_s0_bundle(
                    copied / "directed_role_pairs.csv",
                    copied / "S0_AUDIT.json",
                    copied / "MANIFEST_SHA256.json",
                )

    def _freeze_fixture(self, temporary: Path) -> tuple[dict, Path]:
        checkpoints = []
        for arm in ARM_NAMES:
            for seed in (20260923, 20260924):
                path = temporary / f"{arm}_{seed}.pt"
                torch.save({"experiment": EXPERIMENT, "arm": ARM_NAMES[arm], "seed": seed, "step": 20000, "parameter_count": 1 if arm == "igm" else 2, "state_dict": {}}, path)
                checkpoints.append({"arm": arm, "seed": seed, "step": 20000, "checkpoint": str(path), "checkpoint_sha256": sha256_file(path)})
        dummy = temporary / "frozen_config.txt"
        dummy.write_text("frozen\n", encoding="utf-8")
        official_manifest = temporary / "official_source.sha256"
        official_files = [self.official_source / "default_config.py"]
        official_files.extend((self.official_source / "src").rglob("*.py"))
        official_files.extend((self.official_source / "configs").rglob("*.py"))
        official_manifest.write_text("".join(f"{sha256_file(path)}  {path.resolve()}\n" for path in sorted(official_files)), encoding="utf-8")
        paths = {role: dummy for role in REQUIRED_FREEZE_BINDINGS}
        paths.update(
            {
                "sequence_roles": self.s0 / "sequence_roles.csv",
                "file_manifest": self.s0 / "file_manifest.csv",
                "native_pairs": self.s0 / "native_pairs.csv",
                "directed_role_pairs": self.s0 / "directed_role_pairs.csv",
                "s0_audit": self.s0 / "S0_AUDIT.json",
                "s0_hashes": self.s0 / "MANIFEST_SHA256.json",
                "evaluator_source": self.source / "tools/prompt37_igm_hp1_runner.py",
                "common_source": self.source / "tools/prompt37_igm_hp1_common.py",
                "igm_model_source": self.source / "tools/igm_m0/model.py",
                "igm_loss_source": self.source / "tools/igm_m0/losses.py",
                "official_adapter_source": self.source / "tools/official_slim_hp_scratch_adapter.py",
                "official_source_manifest": official_manifest,
                "hpatches_archive": self.data_archive,
            }
        )
        freeze = {
            "experiment": EXPERIMENT,
            "decision": "EVALUATE_FINAL_ENDPOINTS",
            "bindings": {role: {"path": str(path.resolve()), "sha256": sha256_file(path)} for role, path in paths.items()},
            "endpoints": checkpoints,
        }
        freeze_path = temporary / "endpoint_freeze.json"
        freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
        return freeze, freeze_path

    def _validate_fixture(self, freeze_path: Path, checkpoint: Path) -> None:
        validate_endpoint_freeze(
            freeze_path,
            checkpoint,
            "igm",
            20260923,
            self.s0 / "directed_role_pairs.csv",
            self.s0 / "S0_AUDIT.json",
            self.s0 / "MANIFEST_SHA256.json",
            self.data_root,
            self.data_archive,
            self.official_source,
            Path(json.loads(freeze_path.read_text(encoding="utf-8"))["bindings"]["official_source_manifest"]["path"]),
        )

    def test_endpoint_freeze_negative_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            freeze, freeze_path = self._freeze_fixture(temporary)
            selected = Path(freeze["endpoints"][0]["checkpoint"])
            self._validate_fixture(freeze_path, selected)

            missing = copy.deepcopy(freeze)
            missing["endpoints"].pop()
            freeze_path.write_text(json.dumps(missing), encoding="utf-8")
            touched = []
            original_sha = sha256_file
            with mock.patch("tools.prompt37_igm_hp1_runner.sha256_file", side_effect=lambda path: touched.append(Path(path).resolve()) or original_sha(Path(path))):
                with self.assertRaises(RuntimeError):
                    self._validate_fixture(freeze_path, selected)
            self.assertNotIn((self.s0 / "directed_role_pairs.csv").resolve(), touched)

            wrong_step = copy.deepcopy(freeze)
            wrong_step["endpoints"][0]["step"] = 19999
            freeze_path.write_text(json.dumps(wrong_step), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                self._validate_fixture(freeze_path, selected)

            freeze, freeze_path = self._freeze_fixture(temporary)
            selected = Path(freeze["endpoints"][0]["checkpoint"])
            freeze["bindings"]["common_source"] = freeze["bindings"]["method_spec"]
            freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                self._validate_fixture(freeze_path, selected)

            wrong_metadata = copy.deepcopy(freeze)
            wrong_checkpoint = Path(wrong_metadata["endpoints"][0]["checkpoint"])
            torch.save({"experiment": EXPERIMENT, "arm": ARM_NAMES["official"], "seed": 20260923, "step": 20000, "parameter_count": 2, "state_dict": {}}, wrong_checkpoint)
            wrong_metadata["endpoints"][0]["checkpoint_sha256"] = sha256_file(wrong_checkpoint)
            freeze_path.write_text(json.dumps(wrong_metadata), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                self._validate_fixture(freeze_path, selected)

            freeze, freeze_path = self._freeze_fixture(temporary)
            selected = Path(freeze["endpoints"][0]["checkpoint"])
            freeze["bindings"]["evaluator_source"]["sha256"] = "0" * 64
            freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                self._validate_fixture(freeze_path, selected)

    def test_joint_pairing_bootstrap_and_latency_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            freeze, freeze_path = self._freeze_fixture(temporary)
            endpoint_map = {(entry["arm"], entry["seed"]): entry for entry in freeze["endpoints"]}
            directories = []
            for arm in ARM_NAMES:
                for seed in (20260923, 20260924):
                    directory = temporary / f"{arm}_{seed}"
                    directory.mkdir()
                    summary = {
                        "arm_key": arm,
                        "seed": seed,
                        "inference_seconds_p95": 0.15 if arm == "igm" else 0.2,
                        "parameter_count": 1 if arm == "igm" else 2,
                        "checkpoint_sha256": endpoint_map[(arm, seed)]["checkpoint_sha256"],
                    }
                    with (directory / "per_pair_repeat.csv").open("w", encoding="utf-8") as handle:
                        handle.write("pair_id,repeat\n")
                        for repeat in (66, 67, 68, 69, 70, 71):
                            handle.write(f"p,{repeat}\n")
                    with (directory / "per_sequence_repeat.csv").open("w", encoding="utf-8") as handle:
                        handle.write("sequence,repeat,auc@5\n")
                        for repeat in (66, 67, 68, 69, 70, 71):
                            for sequence in ("i_a", "v_b"):
                                handle.write(f"{sequence},{repeat},{0.6 if arm == 'igm' else 0.5}\n")
                    (directory / "aggregate_repeats.csv").write_text("repeat,sequence_macro_auc@5\n66,0.5\n", encoding="utf-8")
                    latency = 0.15 if arm == "igm" else 0.2
                    (directory / "inference_latency.csv").write_text(f"pair_id,inference_seconds\np,{latency}\n", encoding="utf-8")
                    summary["artifact_hashes"] = {name: sha256_file(directory / name) for name in ("per_pair_repeat.csv", "per_sequence_repeat.csv", "aggregate_repeats.csv", "inference_latency.csv")}
                    (directory / "EVALUATION_SUMMARY.json").write_text(json.dumps(summary), encoding="utf-8")
                    directories.append(directory)
            output = temporary / "joint.json"
            run_aggregate(SimpleNamespace(evaluation_dir=directories, endpoint_freeze=freeze_path, output=output))
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(result["pair_sequence_repeat_keys_exactly_aligned"])
            self.assertAlmostEqual(result["mean_auc5_delta_pp"], 10.0, places=6)
            self.assertAlmostEqual(result["inference_p95_ratio"], 0.75, places=6)
            tampered_summary_path = directories[0] / "EVALUATION_SUMMARY.json"
            tampered_summary = json.loads(tampered_summary_path.read_text(encoding="utf-8"))
            tampered_summary["inference_seconds_p95"] = 999999.0
            tampered_summary["parameter_count"] = 999999999
            tampered_summary_path.write_text(json.dumps(tampered_summary), encoding="utf-8")
            tampered_output = temporary / "summary_tamper.json"
            run_aggregate(SimpleNamespace(evaluation_dir=directories, endpoint_freeze=freeze_path, output=tampered_output))
            tampered_result = json.loads(tampered_output.read_text(encoding="utf-8"))
            self.assertEqual(result["inference_p95_ratio"], tampered_result["inference_p95_ratio"])
            self.assertEqual(result["parameter_count_ratio"], tampered_result["parameter_count_ratio"])
            (directories[0] / "per_pair_repeat.csv").write_text("mutation", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                run_aggregate(SimpleNamespace(evaluation_dir=directories, endpoint_freeze=freeze_path, output=temporary / "bad.json"))

    def test_mutated_evaluation_payload_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            data = temporary / "data"
            data.mkdir()
            payload = data / "i_x.ppm"
            payload.write_bytes(b"original")
            manifest = temporary / "files.csv"
            with manifest.open("w", encoding="utf-8") as handle:
                handle.write("role,relative_path,size_bytes,sha256\n")
                digest = sha256_file(payload)
                for _ in range(18 * 11):
                    handle.write(f"evaluation,i_x.ppm,8,{digest}\n")
            verify_evaluation_payload_bytes(data, manifest)
            payload.write_bytes(b"mutated!")
            with self.assertRaises(RuntimeError):
                verify_evaluation_payload_bytes(data, manifest)

if __name__ == "__main__":
    unittest.main(verbosity=2)
