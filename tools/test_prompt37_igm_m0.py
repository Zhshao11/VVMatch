from __future__ import annotations

import ast
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.prompt37_igm_hp1_common import PairSample, _transform_matrix, restore_points, warp_points
from tools.igm_m0.losses import _directional_coarse_loss, compute_igm_loss
from tools.igm_m0.model import IndependentGeometryMatcherM0
from tools.prompt37_igm_hp1_runner import ARM_NAMES, EXPERIMENT, REQUIRED_FREEZE_BINDINGS, sha256_file, validate_endpoint_freeze


class IGMContractTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = IndependentGeometryMatcherM0().to(self.device)

    def test_shapes_assignment_and_gradient_path(self) -> None:
        image0 = torch.rand(1, 1, 64, 80, device=self.device)
        image1 = torch.rand(1, 1, 64, 80, device=self.device)
        mask = torch.ones(1, 64, 80, dtype=torch.bool, device=self.device)
        output = self.model(image0, image1, mask, mask, return_matches=False)
        self.assertEqual(output["assignment"].shape, (1, 80, 81))
        self.assertLess(float((output["assignment"].sum(-1) - 1.0).abs().max().detach()), 1e-5)
        self.assertTrue(torch.isfinite(output["assignment"]).all())
        sample = PairSample(
            image0=image0,
            image1=image1,
            mask0=mask,
            mask1=mask,
            homography=torch.eye(3, device=self.device)[None],
            original_homography=torch.eye(3, device=self.device)[None],
            original_to_network0=torch.eye(3, device=self.device)[None],
            original_to_network1=torch.eye(3, device=self.device)[None],
            valid_hw0=(64, 80),
            valid_hw1=(64, 80),
            metadata={},
        )
        loss, _ = compute_igm_loss(self.model, output, sample)
        loss.backward()
        groups = {
            "encoder_and_fpn": self.model.encoder.parameters(),
            "dense_pair_interaction": self.model.interaction.parameters(),
            "affinity_and_matchability": list(self.model.source_projection.parameters())
            + list(self.model.target_projection.parameters())
            + list(self.model.source_matchability.parameters())
            + list(self.model.target_matchability.parameters()),
            "bilateral_fine_projection": self.model.fine_projection.parameters(),
        }
        for name, parameters in groups.items():
            gradients = [p.grad for p in parameters if p.requires_grad]
            self.assertTrue(gradients, name)
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients), name)
            self.assertTrue(all(float(g.norm()) > 0.0 for g in gradients), name)

    def test_coordinate_identity_round_trip(self) -> None:
        points = torch.tensor([[0.5, 0.5], [17.25, 31.75], [79.0, 63.0]])
        transform = torch.tensor([[[1.5, 0.0, 7.0], [0.0, 1.25, 5.0], [0.0, 0.0, 1.0]]])
        network = warp_points(points[None], transform)[0]
        restored = restore_points(network, transform)
        # Float32 inverse/forward transforms accumulate about 1.5e-4 px here.
        self.assertLess(float((restored - points).abs().max()), 2e-4)

    def test_target_permutation_changes_action(self) -> None:
        image0 = torch.rand(1, 1, 64, 80, device=self.device)
        image1 = torch.rand(1, 1, 64, 80, device=self.device)
        mask = torch.ones(1, 64, 80, dtype=torch.bool, device=self.device)
        original = self.model(image0, image1, mask, mask, return_matches=False)
        permuted = self.model(image0, image1.flip(-1), mask, mask, return_matches=False)
        self.assertGreater(float((original["pair"] - permuted["pair"]).abs().max().detach()), 1e-8)
        self.assertGreater(self.model.branch_counts["interaction"], 0)

    def test_flip_uses_pixel_center_geometry(self) -> None:
        transform, _ = _transform_matrix((80, 100), 1.0, 0.0, 0.0, True)
        left_center = np.array([0.5, 35.5, 1.0])
        np.testing.assert_allclose(transform @ left_center, np.array([597.0, 213.0, 1.0]), atol=1e-6)

    def test_coarse_mask_uses_token_centers(self) -> None:
        mask = torch.zeros(1, 16, 16, dtype=torch.bool, device=self.device)
        mask[:, :9, :9] = True
        coarse = self.model.coarse_center_mask(mask, 2, 2)
        expected = torch.tensor([[[True, False], [False, False]]], device=self.device)
        self.assertTrue(torch.equal(coarse, expected))

    def test_null_mutual_cap_and_stable_dedup(self) -> None:
        pair = torch.zeros(1, 3, 3, device=self.device)
        pair[0].fill_diagonal_(0.8)
        source_null = torch.full((1, 3, 1), 0.1, device=self.device)
        target_null = torch.full((1, 1, 3), 0.1, device=self.device)
        source_null[0, 1, 0] = 0.9
        _, source_index, target_index, _ = self.model.mutual_matches(pair, source_null, target_null)
        self.assertEqual(source_index.tolist(), [0, 2])
        self.assertEqual(target_index.tolist(), [0, 2])
        count = 2050
        diagonal = torch.eye(count, device=self.device)[None]
        null = torch.zeros(1, count, 1, device=self.device)
        _, source_index, _, _ = self.model.mutual_matches(diagonal, null, null.transpose(1, 2))
        self.assertEqual(source_index.numel(), 2048)
        source = torch.tensor([[4.0, 4.0], [4.0, 4.0], [12.0, 4.0], [-1.0, 4.0]], device=self.device)
        target = torch.tensor([[5.0, 4.0], [6.0, 4.0], [5.0, 4.0], [7.0, 4.0]], device=self.device)
        confidence = torch.tensor([0.7, 0.9, 0.8, 1.0], device=self.device)
        ids = torch.arange(4, device=self.device)
        filtered = self.model.stable_filter_and_deduplicate(source, target, confidence, ids, ids, (16, 16), (16, 16))
        self.assertEqual(filtered[2].numel(), 2)
        self.assertAlmostEqual(float(filtered[2][0]), 0.9, places=5)
        self.assertAlmostEqual(float(filtered[2][1]), 0.8, places=5)

    def test_refinement_masks_edges_and_moves_both_sides(self) -> None:
        fine0 = torch.rand(1, 64, 8, 8, device=self.device)
        fine1 = torch.rand(1, 64, 8, 8, device=self.device)
        source = torch.tensor([[2.0, 2.0]], device=self.device)
        target = torch.tensor([[26.0, 26.0]], device=self.device)
        refined0, refined1, _, _ = self.model.refine(
            fine0, fine1, torch.zeros(1, dtype=torch.long, device=self.device), source, target, (32, 32), (32, 32)
        )
        self.assertTrue(((refined0 >= 0) & (refined0 < 32)).all())
        self.assertTrue(((refined1 >= 0) & (refined1 < 32)).all())
        self.assertGreater(float((refined0 - source).abs().max().detach()), 0.0)
        self.assertGreater(float((refined1 - target).abs().max().detach()), 0.0)

    def test_coarse_nll_matched_unmatched_and_mixed(self) -> None:
        identity = torch.eye(3, device=self.device)[None]
        pair = torch.tensor([[[0.8, 0.0], [0.0, 0.0]]], device=self.device)
        null = torch.tensor([[[0.2], [0.6]]], device=self.device)
        source_mask = torch.tensor([[True, True]], device=self.device)
        target_mask = torch.tensor([[True, False]], device=self.device)
        shifted = identity.clone()
        shifted[:, 0, 2] = 1.0
        mixed, details = _directional_coarse_loss(pair, null, source_mask, target_mask, shifted, (1, 2), (1, 2), (8, 16))
        self.assertEqual((details["matched"], details["unmatched"]), (1, 1))
        self.assertAlmostEqual(float(mixed.detach()), -math.log(0.8) - math.log(0.6), places=5)
        matched, _ = _directional_coarse_loss(pair[:, :1, :1], null[:, :1], source_mask[:, :1], target_mask[:, :1], identity, (1, 1), (1, 1), (8, 8))
        self.assertAlmostEqual(float(matched.detach()), -math.log(0.8), places=5)
        translated = identity.clone()
        translated[:, 0, 2] = 9.0
        unmatched, _ = _directional_coarse_loss(pair[:, 1:, :], null[:, 1:], source_mask[:, 1:], target_mask, translated, (1, 1), (1, 2), (8, 16))
        self.assertAlmostEqual(float(unmatched.detach()), -math.log(0.6), places=5)

    def test_full_identity_action_path(self) -> None:
        with torch.no_grad():
            self.model.target_projection.weight.copy_(self.model.source_projection.weight)
            self.model.source_matchability.weight.zero_()
            self.model.target_matchability.weight.zero_()
            self.model.source_matchability.bias.fill_(20.0)
            self.model.target_matchability.bias.fill_(20.0)
        image = torch.rand(1, 1, 64, 80, device=self.device)
        mask = torch.ones(1, 64, 80, dtype=torch.bool, device=self.device)
        output = self.model(image, image, mask, mask, return_matches=False)
        count = output["pair"].shape[1]
        oracle_pair = torch.eye(count, device=self.device)[None]
        oracle_null = torch.zeros(1, count, 1, device=self.device)
        batch, source_index, target_index, _ = self.model.mutual_matches(
            oracle_pair, oracle_null, oracle_null.transpose(1, 2), maximum=16
        )
        centers = self.model._token_centers(8, 10, 8, self.device)
        source, target, _, _ = self.model.refine(
            output["fine0"], output["fine1"], batch, centers[source_index], centers[target_index], (64, 80), (64, 80)
        )
        self.assertGreater(source.shape[0], 0)
        self.assertLess(float((source - target).abs().max().detach()), 1e-4)

    def test_independent_import_graph_and_state_keys(self) -> None:
        package = ROOT / "tools/igm_m0"
        forbidden = ("official" + "slim", "d0", "jcga")
        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name.lower() for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append((node.module or "").lower())
            self.assertFalse(any(token in item for token in forbidden for item in imports), path)
        keys = [key.lower() for key in self.model.state_dict()]
        self.assertFalse(any(token in key for token in forbidden for key in keys))


if __name__ == "__main__":
    unittest.main(verbosity=2)
