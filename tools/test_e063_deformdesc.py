from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch

from tools.e063_deformdesc import DeformDescBranch


ROOT = Path(__file__).resolve().parents[1]


class DeformDescBranchTests(unittest.TestCase):
    def test_zero_initialized_action_is_exact_identity(self) -> None:
        torch.manual_seed(63)
        module = DeformDescBranch(max_channels=32, rank=8, max_scales=2)
        feature = torch.randn(2, 16, 12, 10)
        output = module(feature, 0)
        self.assertTrue(torch.equal(output, feature))
        self.assertEqual(tuple(module.last_offsets.shape), (2, 12, 10, 2))
        self.assertEqual(tuple(module.last_reliability.shape), (2, 12, 10, 1))

    def test_enabled_action_is_finite_bounded_and_trainable(self) -> None:
        torch.manual_seed(6301)
        module = DeformDescBranch(max_channels=32, rank=8, max_scales=2)
        with torch.no_grad():
            module.mix[1] = 0.5
            module.offset_scale[1] = 0.5
        feature = torch.randn(1, 24, 9, 11, requires_grad=True)
        output = module(feature, 1)
        self.assertEqual(output.shape, feature.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertLessEqual(float(module.last_offsets.abs().max().detach()), 1.0)
        self.assertGreaterEqual(float(module.last_reliability.min().detach()), 0.0)
        self.assertLessEqual(float(module.last_reliability.max().detach()), 1.0)
        self.assertGreater(float((output - feature).abs().max().detach()), 0.0)

        output.square().mean().backward()
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.norm()), 0.0, name)

    def test_invalid_channel_and_scale_are_rejected(self) -> None:
        module = DeformDescBranch(max_channels=8, rank=4, max_scales=1)
        with self.assertRaises(RuntimeError):
            module(torch.randn(1, 9, 4, 4), 0)
        with self.assertRaises(IndexError):
            module(torch.randn(1, 8, 4, 4), 1)

    def test_portable_module_has_no_slim_runtime_dependency(self) -> None:
        path = ROOT / "tools/e063_deformdesc/model.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.lower() for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append((node.module or "").lower())
        for token in ("slim", "ascs", "jcga"):
            self.assertFalse(any(token in item for item in imports), token)


if __name__ == "__main__":
    unittest.main(verbosity=2)
