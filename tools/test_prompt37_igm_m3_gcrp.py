from __future__ import annotations

import ast
import csv
import json
import unittest
from pathlib import Path

import torch

from tools.igm_m0r1 import IndependentGeometryMatcherM0R1
from tools.igm_m3_gcrp import GCRPPretrainer
from tools.igm_m3_gcrp.losses import compute_gcrp_loss
from tools.prompt37_igm_m3_hp3_runner import HP3_BUNDLE, scale_masks, sha


ROOT = Path(__file__).resolve().parents[1]
S0 = ROOT / "research/results/P37-20260930-IGM-M3-HP3/s0_config_audit"


class GCRPTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260930)
        self.model = GCRPPretrainer()

    def test_hp3_split_and_probe_seal(self) -> None:
        self.assertEqual(sha(S0 / "MANIFEST_SHA256.json"), HP3_BUNDLE)
        audit = json.loads((S0 / "S0_AUDIT.json").read_text())
        self.assertEqual(audit["status"], "S0_PASSED_PROBE_DEV_EVAL_UNREAD")
        self.assertTrue(all(value for key, value in audit["checks"].items() if not key.endswith("_read")))
        self.assertFalse(audit["checks"]["probe_payload_read"])
        with (S0 / "pretrain_sequence_roles.csv").open(newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(sum(row["pretrain_role"] == "fit" for row in rows), 64)
        self.assertEqual(sum(row["pretrain_role"] == "probe" for row in rows), 8)

    def test_export_is_exact_m0r1_encoder(self) -> None:
        parent = IndependentGeometryMatcherM0R1()
        exported = self.model.export_encoder()
        self.assertEqual(exported.keys(), parent.encoder.state_dict().keys())
        parent.encoder.load_state_dict(exported, strict=True)
        self.assertFalse(any(name.startswith(("projector", "predictor", "ema", "scale_embedding")) for name in parent.state_dict()))

    def test_loss_branches_and_gradients(self) -> None:
        image0 = torch.rand(1, 1, 64, 80)
        image1 = torch.rand_like(image0)
        token_mask = torch.zeros(1, 8, 10, dtype=torch.bool); token_mask[:, :4, :4] = True
        output = self.model(image0, image1, token_mask, token_mask)
        sample = type("Sample", (), {"mask0": torch.ones(1, 64, 80, dtype=torch.bool), "mask1": torch.ones(1, 64, 80, dtype=torch.bool)})()
        valid0, valid1 = scale_masks(output, sample)
        identity = torch.eye(3)[None]
        loss, metrics = compute_gcrp_loss(self.model, output, identity, valid0, valid1, 20260930)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(metrics[key] >= 0 for key in ("loss_mask", "loss_corr", "loss_geo", "loss_vc")))
        loss.backward()
        missing = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all() or float(parameter.grad.norm()) == 0)]
        self.assertEqual(missing, [])
        self.assertTrue(all(parameter.grad is None for parameter in self.model.ema_encoder.parameters()))

    def test_shuffled_geometry_changes_geometry_loss(self) -> None:
        image0 = torch.rand(1, 1, 64, 80)
        image1 = torch.rand_like(image0)
        token_mask = torch.zeros(1, 8, 10, dtype=torch.bool); token_mask[:, :4, :4] = True
        output = self.model(image0, image1, token_mask, token_mask)
        sample = type("Sample", (), {"mask0": torch.ones(1, 64, 80, dtype=torch.bool), "mask1": torch.ones(1, 64, 80, dtype=torch.bool)})()
        valid0, valid1 = scale_masks(output, sample)
        identity = torch.eye(3)[None]
        shifted = identity.clone(); shifted[:, 0, 2] = 16
        _, correct = compute_gcrp_loss(self.model, output, identity, valid0, valid1, 1)
        _, shuffled = compute_gcrp_loss(self.model, output, shifted, valid0, valid1, 1)
        self.assertNotEqual(correct["loss_geo"], shuffled["loss_geo"])

    def test_forbidden_dependencies_absent(self) -> None:
        files = list((ROOT / "tools/igm_m3_gcrp").glob("*.py")) + [ROOT / "tools/prompt37_igm_m3_hp3_runner.py"]
        source = "\n".join(path.read_text().lower() for path in files)
        for token in ("dinov2", "mast3r", "vggt", "depth anything", "megadepth", "igm_m1_vasi", "ransac_loss", "hard_gate"):
            self.assertNotIn(token, source)
        for path in files:
            ast.parse(path.read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
