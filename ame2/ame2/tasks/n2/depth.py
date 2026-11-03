# SPDX-License-Identifier: BSD-3-Clause
"""深度相机标定和部署预处理，不依赖 Isaac Sim。"""

import torch

CAMERA_POSITION = (0.06387, 0.01629, -0.11057)
CAMERA_ROTATION = (0.2705981, -0.6532815, 0.6532815, -0.2705981)  # 四元数顺序 wxyz，采用 ROS 相机坐标约定。
CAMERA_INTRINSICS = (12.6454, 0.0, 12.0, 0.0, 11.2403, 8.0, 0.0, 0.0, 1.0)
DEPTH_HEIGHT = 16
DEPTH_WIDTH = 24
DEPTH_RAY_RANGE = 4.0
DEPTH_MIN_RANGE = 0.3
DEPTH_MAX_RANGE = 2.0
DEPTH_UPDATE_PERIOD = 0.1  # 采用 AME2 感知侧的更新周期，与 50 Hz 控制独立。
PROPRIO_HISTORY = 5


def preprocess_depth(depth: torch.Tensor, min_range: float = 0.3, max_range: float = 2.0) -> torch.Tensor:
    """输入光轴深度，单位为米、形状为 BCHW；过滤无效值后整体减 1。

    参考 Perceptive-locomotion 的 noetix_n2 DepthCfg/raycaster_depth。
    相机射线可达 4 m，但实际启用的观测显式限制为 2 m。
    无效、过近和过远的像素在处理后均为 -1。
    实机深度也使用同一处理流程；径向距离和光轴深度含义不同。
    """
    valid = torch.isfinite(depth) & (depth >= min_range) & (depth <= max_range)
    return torch.where(valid, depth, torch.zeros_like(depth)) - 1.0
