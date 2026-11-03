# SPDX-License-Identifier: BSD-3-Clause
"""导出 N2 深度 Actor，包含原始深度预处理及部署元数据。"""

import copy
import json
from pathlib import Path

import torch
from torch import nn

from .depth import (
    CAMERA_INTRINSICS, CAMERA_POSITION, CAMERA_ROTATION, DEPTH_HEIGHT, DEPTH_MAX_RANGE,
    DEPTH_MIN_RANGE, DEPTH_WIDTH, PROPRIO_HISTORY, preprocess_depth,
)


class N2DepthPolicy(nn.Module):
    """部署策略输入：本体历史 (B,300)、指令 (B,4)、光轴深度 (B,1,16,24)。"""

    def __init__(self, model):
        super().__init__()
        # 内置 CNN 的 JIT 包装器与训练 Actor 共享 CNN 模块。
        # 移到 CPU 前先复制，保证导出不会改变原 Actor 所在的 GPU。
        self.policy = copy.deepcopy(model.as_jit())

    def forward(self, proprio: torch.Tensor, commands: torch.Tensor, depth_m: torch.Tensor) -> torch.Tensor:
        flat = torch.cat([proprio, commands], dim=-1)
        depth = preprocess_depth(depth_m)
        return self.policy(flat, [depth])

    @torch.jit.export
    def reset(self) -> None:
        # 本体历史和最近一帧异步深度由部署调用端维护。
        pass


def export_policy(model, env, directory: str | Path) -> Path:
    """保存 CPU TorchScript 策略，以及实际仿真关节顺序和 PD 配置。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    robot = env.unwrapped.scene["robot"]
    action = env.unwrapped.action_manager.get_term("joint_pos")
    policy = N2DepthPolicy(model).cpu().eval()
    scripted = torch.jit.freeze(torch.jit.script(policy))
    path = directory / "n2_depth_policy.jit"
    scripted.save(str(path))

    action_ids = action._joint_ids
    if isinstance(action_ids, slice):
        action_ids = list(range(robot.num_joints))[action_ids]
    else:
        action_ids = list(action_ids)
    metadata = {
        "task": "Ame2-N2-Depth",
        "inputs": {"proprio": [1, 300], "commands": [1, 4], "depth_m": [1, 1, DEPTH_HEIGHT, DEPTH_WIDTH]},
        "outputs": {"actions": [1, 18]},
        "policy_period_s": env.unwrapped.step_dt,
        "camera_period_s": env.unwrapped.scene["depth_camera"].cfg.update_period,
        "camera_position_base_m": CAMERA_POSITION,
        "camera_rotation_ros_wxyz": CAMERA_ROTATION,
        "camera_intrinsics": CAMERA_INTRINSICS,
        "depth_type": "optical-axis distance_to_image_plane in metres",
        "depth_valid_range_m": [DEPTH_MIN_RANGE, DEPTH_MAX_RANGE],
        "depth_preprocessing": "included in JIT: invalid/near/far -> 0, then depth - 1",
        "proprio_history_frames": PROPRIO_HISTORY,
        "proprio_history_order": "oldest to newest; repeat first frame on reset",
        "proprio_frame_order": ["base_angular_velocity_body", "projected_gravity", "joint_pos_minus_default",
                                 "joint_velocity", "last_applied_action"],
        "commands_order": ["goal_x_body", "goal_y_body", "sin_heading_error", "cos_heading_error"],
        "commands_position_clip_m": 2.0,
        "joint_names": list(robot.joint_names),
        "action_joint_names": [robot.joint_names[i] for i in action_ids],
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
        "action_scale": action._scale[0].cpu().tolist(),
        "action_offset": action._offset[0].cpu().tolist(),
        "joint_stiffness": robot.data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": robot.data.joint_damping[0].cpu().tolist(),
        "joint_effort_limits": robot.data.joint_effort_limits[0].cpu().tolist(),
        "joint_velocity_limits": robot.data.joint_vel_limits[0].cpu().tolist(),
    }
    (directory / "n2_depth_policy.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return path
