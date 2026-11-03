# SPDX-License-Identifier: BSD-3-Clause
"""N2 部署观测，Actor 不接收特权状态。"""

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .. import mdp
from .depth import DEPTH_MAX_RANGE, DEPTH_MIN_RANGE, preprocess_depth


def proprioception(env, noisy: bool = True) -> torch.Tensor:
    """单帧 60 维：角速度、重力投影、相对关节位置、相对关节速度和上一动作。"""
    values = [
        mdp.ame2_base_ang_vel(env),
        mdp.ame2_projected_gravity(env),
        mdp.ame2_joint_pos_rel(env),
        mdp.ame2_joint_vel_rel(env),
        env.action_manager.action.clamp(-50.0, 50.0),  # 观测时最近一条已施加的策略动作。
    ]
    if noisy:
        # 沿用 AME2 学生策略的本体状态噪声幅度。
        values = [value + (2 * torch.rand_like(value) - 1) * amplitude
                  for value, amplitude in zip(values, (0.2, 0.05, 0.01, 1.5, 0.0))]
    return torch.cat(values, dim=-1)


class DepthImage(ManagerTermBase):
    """两次传感器更新之间保持最近一帧图像及其噪声。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        camera = env.scene[cfg.params["sensor_cfg"].name]
        height, width = camera.image_shape
        self._last_frame = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self._depth = torch.full((env.num_envs, 1, height, width), -1.0, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._last_frame[env_ids] = -1
        self._depth[env_ids] = -1.0

    def __call__(
        self,
        env,
        sensor_cfg: SceneEntityCfg,
        noisy: bool = True,
        min_range: float = DEPTH_MIN_RANGE,
        max_range: float = DEPTH_MAX_RANGE,
    ) -> torch.Tensor:
        camera = env.scene[sensor_cfg.name]
        raw = camera.data.output["distance_to_image_plane"]  # 输出为 BHWC；访问数据时更新到期的传感器缓存。
        frame = camera.frame
        due = frame != self._last_frame
        if due.any():
            depth = raw[due].permute(0, 3, 1, 2).clone()
            if noisy:
                # 参考 N2 任务中的随距离变化噪声和高斯噪声。
                depth += (2 * torch.rand_like(depth) - 1) * (0.01 * depth)
                depth += torch.randn_like(depth) * (0.005 * depth)
                random = torch.rand_like(depth)
                depth = torch.where(random < 0.005, torch.zeros_like(depth), depth)
                depth = torch.where((random >= 0.005) & (random < 0.01), max_range, depth)
            self._depth[due] = preprocess_depth(depth, min_range, max_range)
            self._last_frame[due] = frame[due]
        return self._depth
