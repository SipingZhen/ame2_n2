# SPDX-License-Identifier: BSD-3-Clause
"""在 CPU 上测试实机深度预处理约定，无需启动 Isaac Sim。"""

import importlib.util
from pathlib import Path
import unittest

import torch

path = Path(__file__).resolve().parents[2] / "ame2/tasks/n2/depth.py"
spec = importlib.util.spec_from_file_location("n2_depth_preprocessing", path)
depth_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(depth_module)


class TestDepthPreprocessing(unittest.TestCase):
    def test_invalid_and_boundary_depths(self):
        raw = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0,
                            0.299, 0.3, 1.0, 2.0, 2.001]).reshape(1, 1, 1, -1)
        original = raw.clone()
        expected = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -0.7, 0.0, 1.0, -1.0]).reshape_as(raw)
        processed = depth_module.preprocess_depth(raw)
        torch.testing.assert_close(processed, expected)
        torch.testing.assert_close(raw, original, equal_nan=True)
        self.assertTrue(torch.isfinite(processed).all())

    def test_deployment_torchscript_matches_python(self):
        raw = torch.rand(3, 1, 16, 24) * 4.0
        scripted = torch.jit.script(depth_module.preprocess_depth)
        torch.testing.assert_close(scripted(raw), depth_module.preprocess_depth(raw))


if __name__ == "__main__":
    unittest.main()
