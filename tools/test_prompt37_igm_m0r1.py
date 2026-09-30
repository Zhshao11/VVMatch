from __future__ import annotations

import ast
import copy
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.igm_m0 import IndependentGeometryMatcherM0
from tools.igm_m0r1 import IndependentGeometryMatcherM0R1


class M0R1ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260923)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = IndependentGeometryMatcherM0R1().to(self.device)

    def test_categorical_sums_masks_and_empty_opposite(self) -> None:
        logits = torch.tensor([[[4.0, 1.0, 9.0], [1.0, 5.0, 8.0]]], device=self.device)
        source_null = torch.tensor([[[-2.0], [7.0]]], device=self.device)
        target_null = torch.tensor([[[-3.0], [-3.0], [-3.0]]], device=self.device)
        source_mask = torch.tensor([[True, False]], device=self.device)
        target_mask = torch.tensor([[True, True, False]], device=self.device)
        output = self.model.categorical_assignment(logits, source_null, target_null, source_mask, target_mask)
        self.assertLess(float((output["source_assignment"].sum(-1) - 1).abs().max()), 1e-5)
        self.assertLess(float((output["target_assignment"].sum(-1) - 1).abs().max()), 1e-5)
        self.assertEqual(float(output["source_assignment"][0, 1, -1]), 1.0)
        self.assertEqual(float(output["target_assignment"][0, 2, -1]), 1.0)
        self.assertEqual(float(output["source_assignment"][0, 0, 2]), 0.0)
        all_target_invalid = self.model.categorical_assignment(
            logits, source_null, target_null, source_mask, torch.zeros_like(target_mask)
        )
        self.assertEqual(float(all_target_invalid["source_assignment"][0, 0, -1]), 1.0)

    def test_mutual_non_null_decoder_and_confidence(self) -> None:
        source = torch.tensor([[[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]]], device=self.device)
        target = torch.tensor([[[0.9, 0.05, 0.05], [0.05, 0.1, 0.85]]], device=self.device)
        _, source_id, target_id, confidence = self.model.mutual_matches(source, target)
        self.assertEqual(source_id.tolist(), [0])
        self.assertEqual(target_id.tolist(), [0])
        self.assertAlmostEqual(float(confidence[0]), (0.8 * 0.9) ** 0.5, places=6)
        source[:, :, -1] = 0.99
        _, source_id, _, _ = self.model.mutual_matches(source, target)
        self.assertEqual(source_id.numel(), 0)

    def test_free_logits_fit_identity_unmatched_and_padding(self) -> None:
        logits = torch.zeros(1, 4, 4, device=self.device, requires_grad=True)
        source_null = torch.zeros(1, 4, 1, device=self.device, requires_grad=True)
        target_null = torch.zeros(1, 4, 1, device=self.device, requires_grad=True)
        source_mask = torch.tensor([[True, True, True, False]], device=self.device)
        target_mask = torch.tensor([[True, True, True, False]], device=self.device)
        optimizer = torch.optim.Adam([logits, source_null, target_null], lr=0.3)
        for _ in range(100):
            output = self.model.categorical_assignment(logits, source_null, target_null, source_mask, target_mask)
            source_assignment = output["source_assignment"]
            target_assignment = output["target_assignment"]
            loss = -torch.log(source_assignment[0, 0, 0]) - torch.log(source_assignment[0, 1, 1])
            loss += -torch.log(source_assignment[0, 2, -1])
            loss += -torch.log(target_assignment[0, 0, 0]) - torch.log(target_assignment[0, 1, 1])
            loss += -torch.log(target_assignment[0, 2, -1])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        output = self.model.categorical_assignment(logits, source_null, target_null, source_mask, target_mask)
        _, source_id, target_id, _ = self.model.mutual_matches(output["source_assignment"], output["target_assignment"])
        self.assertEqual(source_id.tolist(), [0, 1])
        self.assertEqual(target_id.tolist(), [0, 1])
        self.assertEqual(int(output["source_assignment"][0, 2].argmax()), 4)
        self.assertEqual(int(output["target_assignment"][0, 2].argmax()), 4)
        self.assertEqual(float(output["source_assignment"][0, 3, -1]), 1.0)

    def test_repeated_texture_free_logits_remain_exact_categorical(self) -> None:
        logits = torch.tensor([[[2.0, 2.0], [2.0, 2.0]]], device=self.device)
        null = torch.full((1, 2, 1), -10.0, device=self.device)
        mask = torch.ones(1, 2, dtype=torch.bool, device=self.device)
        output = self.model.categorical_assignment(logits, null, null, mask, mask)
        self.assertTrue(torch.isfinite(output["pair"]).all())
        self.assertLess(float((output["source_assignment"].sum(-1) - 1).abs().max()), 1e-5)
        self.assertLess(float((output["target_assignment"].sum(-1) - 1).abs().max()), 1e-5)

    def test_parent_initialization_unchanged_outside_assignment_semantics(self) -> None:
        torch.manual_seed(20260923)
        parent = IndependentGeometryMatcherM0().to(self.device)
        torch.manual_seed(20260923)
        rescue = IndependentGeometryMatcherM0R1().to(self.device)
        parent_state = parent.state_dict()
        rescue_state = rescue.state_dict()
        self.assertEqual(parent_state.keys(), rescue_state.keys())
        for key in parent_state:
            self.assertTrue(torch.equal(parent_state[key], rescue_state[key]), key)

    def test_independent_import_graph_and_forbidden_mechanisms(self) -> None:
        package = ROOT / "tools/igm_m0r1"
        forbidden_imports = ("official" + "slim", "jcga")
        forbidden_mechanisms = ("forced_topk", "count_penalty", "confidence_threshold", "covisibility")
        combined = ""
        for path in package.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            combined += source.lower()
            tree = ast.parse(source)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name.lower() for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append((node.module or "").lower())
            self.assertFalse(any(token in item for token in forbidden_imports for item in imports), path)
        self.assertFalse(any(token in combined for token in forbidden_mechanisms))

    def test_every_parameter_receives_gradient(self) -> None:
        image0 = torch.rand(1, 1, 64, 80, device=self.device)
        image1 = torch.rand(1, 1, 64, 80, device=self.device)
        mask = torch.ones(1, 64, 80, dtype=torch.bool, device=self.device)
        output = self.model(image0, image1, mask, mask, return_matches=False)
        loss = output["source_assignment"][:, :, :-1].square().sum()
        loss += output["target_assignment"][:, :, :-1].square().sum()
        centers = self.model._token_centers(8, 10, 8, self.device)[:8]
        batch = torch.zeros(8, dtype=torch.long, device=self.device)
        refined0, refined1, peak0, peak1 = self.model.refine(
            output["fine0"], output["fine1"], batch, centers, centers + 1.0, (64, 80), (64, 80)
        )
        loss += refined0.square().mean() + refined1.square().mean() + peak0.mean() + peak1.mean()
        loss.backward()
        missing = [name for name, parameter in self.model.named_parameters() if parameter.grad is None or not torch.isfinite(parameter.grad).all() or float(parameter.grad.norm()) == 0.0]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
