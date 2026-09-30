from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import torch
from tools.igm_m0r1 import IndependentGeometryMatcherM0R1

from tools.prompt37_igm_m0r1_hp1_runner import (
    ARM_NAMES,
    EXPERIMENT,
    REQUIRED_FREEZE_BINDINGS,
    sha256_file,
    validate_endpoint_freeze,
    validate_s0_bundle,
    validate_stage_settings,
    run_aggregate,
    implementation_bindings,
    validate_tiny_gate,
    validate_conditional_tiny_gate,
    verify_role_payload_bytes,
    verify_evaluation_payload_bytes,
    stable_seed64,
    tiny_rows,
)
from tools.prompt37_igm_hp1_common import deterministic_pair, read_pair_manifest


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
            recorded = json.loads((copied / "MANIFEST_SHA256.json").read_text(encoding="utf-8"))
            recorded["directed_role_pairs.csv"] = sha256_file(copied / "directed_role_pairs.csv")
            (copied / "MANIFEST_SHA256.json").write_text(json.dumps(recorded), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_s0_bundle(copied / "directed_role_pairs.csv", copied / "S0_AUDIT.json", copied / "MANIFEST_SHA256.json")
        with self.assertRaises(RuntimeError):
            validate_conditional_tiny_gate("formal", None)

    def test_launch_guard_anchors_approved_runtime_manifest(self) -> None:
        guard = self.source / "tools/prompt37_igm_m0r1_launch_guard.py"
        approved = self.root / "APPROVED_RUNTIME_SOURCE_SHA256_R7.txt"
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            runner = self.source / "tools/prompt37_igm_m0r1_hp1_runner.py"
            command = [sys.executable, str(guard), "--approved-source-hashes", str(approved), "--python", sys.executable, "--runner", str(runner), "--", "train", "--help", "--approved-source-hashes", str(approved)]
            self.assertEqual(subprocess.run(command, check=False).returncode, 0)
            substitute = temporary / "prompt37_igm_m0r1_hp1_runner.py"; substitute.write_text("pass\n", encoding="utf-8")
            substitute_command = command.copy(); substitute_command[substitute_command.index(str(runner))] = str(substitute)
            self.assertNotEqual(subprocess.run(substitute_command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode, 0)
            mutated = temporary / "mutated.sha256"; mutated.write_text("changed\n", encoding="utf-8")
            command[command.index(str(approved))] = str(mutated)
            self.assertNotEqual(subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode, 0)

    def _freeze_fixture(self, temporary: Path) -> tuple[dict, Path]:
        approved_sources = temporary / "approved_sources.sha256"
        approved_sources.write_text("".join(f"{digest}  {name}\n" for name, digest in implementation_bindings().items()), encoding="utf-8")
        tiny_paths = {role: temporary / f"tiny_{role}.bin" for role in ("checkpoint", "summary", "pair_metrics", "trajectory", "sample_stream", "optimizer_membership", "tensor_gradients", "parameter_deltas")}
        model = IndependentGeometryMatcherM0R1()
        named_parameters = dict(model.named_parameters())
        frozen_tiny = tiny_rows(read_pair_manifest(self.s0 / "directed_role_pairs.csv", "train"))
        sample_rows = []
        for exposure in range(4000):
            row = deterministic_pair(frozen_tiny, 20260923, exposure)
            metadata = {**row, "seed": 20260923, "exposure": exposure, "rng_seed": stable_seed64("augmentation", 20260923, exposure, row["sample_id"])}
            sample_rows.append(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
        tiny_paths["sample_stream"].write_text("\n".join(sample_rows) + "\n", encoding="utf-8")
        sample_hash = sha256_file(tiny_paths["sample_stream"])
        approved_hash = sha256_file(approved_sources)
        provenance = {"s0_bundle_sha256": "9718b1a04da78d061ecee010477f4f16e6d7a3c570caa0d7b61075e745149952", "hpatches_archive_sha256": "99cf7e1ca167896eb4ca2fe3d903beff6566243a3c0c8e07a13a2596648ad670", "data_root": str(self.data_root.resolve()), "sample_stream_sha256": sample_hash, "approved_source_manifest_sha256": approved_hash, "tiny_gate_sha256": None}
        torch.save({"experiment": EXPERIMENT, "arm": "IGM-M0R1", "seed": 20260923, "step": 1000, "parameter_count": sum(parameter.numel() for parameter in named_parameters.values()), "provenance": provenance, "state_dict": model.state_dict()}, tiny_paths["checkpoint"])
        tiny_paths["pair_metrics"].write_text("sample_id,inliers_at_3px,match_count,source_null_argmax_rate,target_null_argmax_rate\n" + "".join(f"{row['sample_id']},20,20,0.0,0.0\n" for row in frozen_tiny), encoding="utf-8")
        tiny_paths["trajectory"].write_text("step,optimizer_steps,processed_samples,loss_total\n" + "".join(f"{step},{step},{step * 4},{1.0 if step <= 20 else 0.2}\n" for step in range(1, 1001)), encoding="utf-8")
        tiny_paths["optimizer_membership"].write_text(json.dumps({"groups": {"g": [{"name": name, "numel": parameter.numel()} for name, parameter in named_parameters.items()]}}), encoding="utf-8")
        tiny_paths["tensor_gradients"].write_text("step,name,gradient_l2,finite_nonzero\n" + "".join(f"1000,{name},1.0,True\n" for name in named_parameters), encoding="utf-8")
        tiny_paths["parameter_deltas"].write_text("name,delta_l2,updated\n" + "".join(f"{name},1.0,True\n" for name in named_parameters), encoding="utf-8")
        checkpoint_hash = sha256_file(tiny_paths["checkpoint"])
        s0_binding = validate_s0_bundle(self.s0 / "directed_role_pairs.csv", self.s0 / "S0_AUDIT.json", self.s0 / "MANIFEST_SHA256.json")
        data_binding = {"data_root": str(self.data_root.resolve()), "archive": str(self.data_archive.resolve()), "archive_sha256": "99cf7e1ca167896eb4ca2fe3d903beff6566243a3c0c8e07a13a2596648ad670"}
        approved_binding = {"path": str(approved_sources.resolve()), "sha256": approved_hash, "files": implementation_bindings()}
        tiny_paths["summary"].write_text(json.dumps({"experiment": EXPERIMENT, "stage": "tiny", "arm": "IGM-M0R1", "seed": 20260923, "steps": 1000, "checkpoint_sha256": checkpoint_hash, "sample_stream_sha256": sample_hash, "initial_loss_mean": 1.0, "final_loss_mean": 0.2, "loss_ratio": 0.2, "checkpoint_reload_exact": True, "s0_binding": s0_binding, "data_binding": data_binding, "approved_source_binding": approved_binding}), encoding="utf-8")
        tiny_files = {role: {"path": str(path.resolve()), "sha256": sha256_file(path)} for role, path in tiny_paths.items()}
        tiny_files.update({
            "directed_role_pairs": {"path": str((self.s0 / "directed_role_pairs.csv").resolve()), "sha256": sha256_file(self.s0 / "directed_role_pairs.csv")},
            "file_manifest": {"path": str((self.s0 / "file_manifest.csv").resolve()), "sha256": sha256_file(self.s0 / "file_manifest.csv")},
            "s0_audit": {"path": str((self.s0 / "S0_AUDIT.json").resolve()), "sha256": sha256_file(self.s0 / "S0_AUDIT.json")},
            "s0_hashes": {"path": str((self.s0 / "MANIFEST_SHA256.json").resolve()), "sha256": sha256_file(self.s0 / "MANIFEST_SHA256.json")},
            "data_archive": {"path": str(self.data_archive.resolve()), "sha256": sha256_file(self.data_archive)},
            "approved_source_hashes": {"path": str(approved_sources.resolve()), "sha256": approved_hash},
        })
        tiny_gate_path = temporary / "TINY_GATE.json"
        gate_values = {"loss_ratio": 0.2, "pairs_ge_20_inliers": 8, "every_pair_ge_20_matches": True, "both_sides_all_null_pairs": 0, "gradient_tensors": True, "update_tensors": True, "reload_exact": True}
        tiny_gate_path.write_text(json.dumps({"experiment": EXPERIMENT, "stage": "tiny", "arm": "IGM-M0R1", "seed": 20260923, "steps": 1000, "passed": True, "gate_values": gate_values, "artifacts": tiny_files}), encoding="utf-8")
        tiny_gate_hash = sha256_file(tiny_gate_path)
        source_manifest_hash = sha256_file(approved_sources)
        checkpoints = []
        for arm in ARM_NAMES:
            for seed in (20260923, 20260924):
                path = temporary / f"{arm}_{seed}.pt"
                provenance = {"s0_bundle_sha256": "9718b1a04da78d061ecee010477f4f16e6d7a3c570caa0d7b61075e745149952", "hpatches_archive_sha256": "99cf7e1ca167896eb4ca2fe3d903beff6566243a3c0c8e07a13a2596648ad670", "data_root": str(self.data_root.resolve()), "sample_stream_sha256": f"stream-{seed}", "approved_source_manifest_sha256": source_manifest_hash, "tiny_gate_sha256": tiny_gate_hash}
                torch.save({"experiment": EXPERIMENT, "arm": ARM_NAMES[arm], "seed": seed, "step": 20000, "parameter_count": 1 if arm == "igm_r1" else 2, "provenance": provenance, "state_dict": {}}, path)
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
                "evaluator_source": self.source / "tools/prompt37_igm_m0r1_hp1_runner.py",
                "common_source": self.source / "tools/prompt37_igm_hp1_common.py",
                "igm_model_source": self.source / "tools/igm_m0r1/model.py",
                "igm_loss_source": self.source / "tools/igm_m0r1/losses.py",
                "official_adapter_source": self.source / "tools/official_slim_hp_scratch_adapter.py",
                "official_source_manifest": official_manifest,
                "hpatches_archive": self.data_archive,
                "tiny_gate": tiny_gate_path,
                "approved_source_hashes": approved_sources,
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
            "igm_r1",
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
            with mock.patch("tools.prompt37_igm_m0r1_hp1_runner.sha256_file", side_effect=lambda path: touched.append(Path(path).resolve()) or original_sha(Path(path))):
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
            official_entry = next(entry for entry in freeze["endpoints"] if entry["arm"] == "official" and entry["seed"] == 20260923)
            official_checkpoint = Path(official_entry["checkpoint"])
            payload = torch.load(official_checkpoint, map_location="cpu", weights_only=False)
            payload["provenance"]["sample_stream_sha256"] = "different"
            torch.save(payload, official_checkpoint)
            official_entry["checkpoint_sha256"] = sha256_file(official_checkpoint)
            freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                self._validate_fixture(freeze_path, selected)

            freeze, freeze_path = self._freeze_fixture(temporary)
            selected = Path(freeze["endpoints"][0]["checkpoint"])
            tiny_path = Path(freeze["bindings"]["tiny_gate"]["path"])
            tiny = json.loads(tiny_path.read_text(encoding="utf-8")); tiny["passed"] = False
            tiny_path.write_text(json.dumps(tiny), encoding="utf-8")
            freeze["bindings"]["tiny_gate"]["sha256"] = sha256_file(tiny_path)
            freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
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
                        "inference_seconds_p95": 0.15 if arm == "igm_r1" else 0.2,
                        "parameter_count": 1 if arm == "igm_r1" else 2,
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
                                handle.write(f"{sequence},{repeat},{0.6 if arm == 'igm_r1' else 0.5}\n")
                    (directory / "aggregate_repeats.csv").write_text("repeat,sequence_macro_auc@5\n66,0.5\n", encoding="utf-8")
                    latency = 0.15 if arm == "igm_r1" else 0.2
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

    def test_mutated_train_payload_and_tiny_artifact_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            data = temporary / "data"; data.mkdir()
            payload = data / "i_x.ppm"; payload.write_bytes(b"original")
            manifest = temporary / "files.csv"
            with manifest.open("w", encoding="utf-8") as handle:
                handle.write("role,relative_path,size_bytes,sha256\n")
                digest = sha256_file(payload)
                for _ in range(72 * 11):
                    handle.write(f"train,i_x.ppm,8,{digest}\n")
            verify_role_payload_bytes(data, manifest, "train", 72 * 11)
            payload.write_bytes(b"mutated!")
            with self.assertRaises(RuntimeError):
                verify_role_payload_bytes(data, manifest, "train", 72 * 11)
            freeze, _ = self._freeze_fixture(temporary)
            tiny_path = Path(freeze["bindings"]["tiny_gate"]["path"])
            validate_tiny_gate(tiny_path)
            tiny = json.loads(tiny_path.read_text(encoding="utf-8"))
            Path(tiny["artifacts"]["summary"]["path"]).write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                validate_tiny_gate(tiny_path)
            freeze, _ = self._freeze_fixture(temporary)
            tiny_path = Path(freeze["bindings"]["tiny_gate"]["path"])
            tiny = json.loads(tiny_path.read_text(encoding="utf-8"))
            pair_path = Path(tiny["artifacts"]["pair_metrics"]["path"])
            rows = pair_path.read_text(encoding="utf-8").splitlines()
            fields = rows[1].split(","); fields[0] = "i_not_the_frozen_pair:1->2"; rows[1] = ",".join(fields)
            pair_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            tiny["artifacts"]["pair_metrics"]["sha256"] = sha256_file(pair_path)
            tiny_path.write_text(json.dumps(tiny), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_tiny_gate(tiny_path)

if __name__ == "__main__":
    unittest.main(verbosity=2)
