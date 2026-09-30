from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.igm_m0r1 import IndependentGeometryMatcherM0R1
from tools.igm_m1_vasi import IndependentGeometryMatcherM1VASI


class VASIContractTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260926)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = IndependentGeometryMatcherM1VASI().to(self.device)

    def _input(self):
        image0 = torch.rand(1, 1, 64, 80, device=self.device)
        image1 = torch.rand(1, 1, 64, 80, device=self.device)
        mask = torch.ones(1, 64, 80, dtype=torch.bool, device=self.device)
        return image0, image1, mask

    def test_route_shapes_sums_spans_and_covisibility(self) -> None:
        image0, image1, mask = self._input()
        output = self.model(image0, image1, mask, mask, return_matches=False)
        self.assertEqual(output["guide_distribution01"].shape, (1, 20, 21))
        self.assertEqual(output["guide_distribution10"].shape, (1, 20, 21))
        self.assertLess(float((output["guide_distribution01"].sum(-1) - 1).abs().max()), 1e-5)
        self.assertLess(float((output["guide_distribution10"].sum(-1) - 1).abs().max()), 1e-5)
        for key in ("route_radius01", "route_radius10"):
            self.assertTrue(torch.isfinite(output[key]).all())
            self.assertGreaterEqual(float(output[key].min()), 2.0)
            self.assertLessEqual(float(output[key].max()), 12.0)
        for key in ("guide_covis01", "guide_covis10"):
            self.assertTrue(((output[key] >= 0) & (output[key] <= 1)).all())

    def test_empty_opposite_grid_maps_to_null_one(self) -> None:
        image0, image1, mask = self._input()
        empty = torch.zeros_like(mask)
        output = self.model(image0, image1, mask, empty, return_matches=False)
        self.assertTrue(torch.equal(output["guide_distribution01"][..., -1], torch.ones_like(output["guide_distribution01"][..., -1])))
        self.assertEqual(float(output["guide_covis01"].max()), 0.0)
        self.assertEqual(float(output["vasi_block0_message01"].abs().max()), 0.0)

    def test_zero_cross_multiplier_and_permutation_controls(self) -> None:
        image0, image1, mask = self._input()
        identity = self.model(image0, image1, mask, mask, return_matches=False, cross_multiplier=0.0)
        self.assertEqual(float(identity["vasi_block0_message01"].abs().max()), 0.0)
        self.assertEqual(float(identity["vasi_block1_message10"].abs().max()), 0.0)
        normal = self.model(image0, image1, mask, mask, return_matches=False)
        perm = torch.randperm(normal["route_mu01_s8"].shape[1], device=self.device)
        override = {"mu01": normal["route_mu01_s8"][:, perm], "mu10": normal["route_mu10_s8"][:, perm]}
        shuffled = self.model(image0, image1, mask, mask, return_matches=False, route_override=override)
        self.assertGreater(float((normal["vasi_block0_message01"] - shuffled["vasi_block0_message01"]).abs().max()), 1e-7)

    def test_cross_attention_uses_post_self_context(self) -> None:
        image0, image1, mask = self._input()
        output = self.model(image0, image1, mask, mask, return_matches=False)
        self.assertEqual(self.model.branch_counts["interaction"], 2)
        changed = self.model(image0, image1.flip(-1), mask, mask, return_matches=False)
        self.assertGreater(float((output["source_state"] - changed["source_state"]).abs().max()), 1e-7)

    def test_inherited_initial_tensors_are_exact(self) -> None:
        torch.manual_seed(20260926); parent = IndependentGeometryMatcherM0R1().to(self.device)
        torch.manual_seed(20260926); child = IndependentGeometryMatcherM1VASI().to(self.device)
        parent_state = parent.state_dict(); child_state = child.state_dict()
        inherited_prefixes = ("encoder.", "source_projection.", "target_projection.", "source_matchability.", "target_matchability.", "fine_projection.", "temperature")
        inherited = [key for key in parent_state if key.startswith(inherited_prefixes)]
        self.assertTrue(inherited)
        for key in inherited:
            self.assertTrue(torch.equal(parent_state[key], child_state[key]), key)

    def test_assignment_and_refiner_semantics_match_parent(self) -> None:
        torch.manual_seed(20260926); parent = IndependentGeometryMatcherM0R1().to(self.device)
        torch.manual_seed(20260926); child = IndependentGeometryMatcherM1VASI().to(self.device)
        source = torch.rand(1, 12, 128, device=self.device); target = torch.rand(1, 10, 128, device=self.device)
        source_mask = torch.ones(1, 12, dtype=torch.bool, device=self.device); target_mask = torch.ones(1, 10, dtype=torch.bool, device=self.device)
        a = parent.assign(source, target, source_mask, target_mask); b = child.assign(source, target, source_mask, target_mask)
        for key in ("source_assignment", "target_assignment", "pair"):
            self.assertTrue(torch.equal(a[key], b[key]), key)
        fine0 = torch.rand(1, 64, 16, 16, device=self.device); fine1 = torch.rand_like(fine0)
        centers = torch.tensor([[30.0, 30.0], [40.0, 44.0]], device=self.device); batch = torch.zeros(2, dtype=torch.long, device=self.device)
        ar = parent.refine(fine0, fine1, batch, centers, centers + 1, (64, 64), (64, 64)); br = child.refine(fine0, fine1, batch, centers, centers + 1, (64, 64), (64, 64))
        for left, right in zip(ar, br): self.assertTrue(torch.equal(left, right))

    def test_every_tensor_receives_gradient(self) -> None:
        image0, image1, mask = self._input()
        output = self.model(image0, image1, mask, mask, return_matches=False)
        loss = output["source_assignment"].square().sum() + output["target_assignment"].square().sum()
        loss += output["guide_distribution01"].square().sum() + output["guide_distribution10"].square().sum()
        loss += output["vasi_block0_message01"].square().mean() + output["vasi_block1_message10"].square().mean()
        centers = self.model._token_centers(8, 10, 8, self.device)[:4]; batch = torch.zeros(4, dtype=torch.long, device=self.device)
        refined = self.model.refine(output["fine0"], output["fine1"], batch, centers, centers + 1, (64, 80), (64, 80))
        loss += sum(value.square().mean() for value in refined)
        loss.backward()
        missing = [name for name, parameter in self.model.named_parameters() if parameter.grad is None or not torch.isfinite(parameter.grad).all() or float(parameter.grad.norm()) == 0]
        self.assertEqual(missing, [])

    def test_forbidden_mechanisms_absent(self) -> None:
        source = "".join(path.read_text(encoding="utf-8").lower() for path in (ROOT / "tools/igm_m1_vasi").glob("*.py"))
        for token in ("count_penalty", "hard_gate", "teacher", "ransac_loss", "m2", "m3"):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
