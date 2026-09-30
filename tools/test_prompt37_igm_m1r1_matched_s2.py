from __future__ import annotations

import json
import unittest
from pathlib import Path

from tools.prompt37_igm_m1r1_matched_s2 import PRIMARY_CASES, compute_gates, file_hash, homographies, synthetic_image


ROOT = Path(__file__).resolve().parents[1]


class MatchedS2Tests(unittest.TestCase):
    def test_frozen_runtime_manifest(self) -> None:
        manifest = json.loads((ROOT / "research/experiments/P37-20260929-IGM-M1R1-HP2_RUNTIME_MANIFEST.json").read_text(encoding="utf-8"))
        paths = {
            "runner": ROOT / "tools/prompt37_igm_m1r1_matched_s2.py",
            "vasi_model": ROOT / "tools/igm_m1_vasi/model.py",
            "vasi_loss": ROOT / "tools/igm_m1_vasi/losses.py",
            "parent_model": ROOT / "tools/igm_m0r1/model.py",
            "parent_loss": ROOT / "tools/igm_m0r1/losses.py",
        }
        self.assertEqual(manifest["runtime_sha256"], {name: file_hash(path) for name, path in paths.items()})
        self.assertEqual(set(manifest["primary_cases"]), PRIMARY_CASES)
        self.assertEqual(manifest["seed"], 20260929)
        self.assertEqual(manifest["steps"], 600)

    def test_four_case_geometry_and_primary_pool(self) -> None:
        cases = homographies(synthetic_image())
        self.assertEqual(list(cases), ["identity", "translation", "scale_rotation", "projective"])
        self.assertEqual(PRIMARY_CASES, {"scale_rotation", "projective"})

    def test_gate_boundaries_and_fail_closed(self) -> None:
        evaluation = {
            "pooled_viewpoint_median_px": {"parent": 2.0, "gt": 1.7, "predicted": 1.9, "permuted": 2.15},
            "viewpoint_route_coverage": {"gt": 0.95, "predicted": 0.90, "permuted": 0.50},
            "all_paths_all_cases_nonempty": True,
        }
        self.assertTrue(all(compute_gates(evaluation, {"x": True}).values()))
        failed = {**evaluation, "pooled_viewpoint_median_px": {**evaluation["pooled_viewpoint_median_px"], "predicted": 1.91}}
        self.assertFalse(compute_gates(failed, {"x": True})["predicted_viewpoint_le_0_95_parent"])
        self.assertFalse(compute_gates(evaluation, {"x": False})["all_integrity_gates"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
