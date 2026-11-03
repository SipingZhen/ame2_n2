# SPDX-License-Identifier: BSD-3-Clause
"""独立的 N2 深度相机任务，保留原有 G1 任务注册。"""

import gymnasium as gym

for task_id, env_class in (
    ("Ame2-N2-Depth", "N2DepthEnvCfg"),
    ("Ame2-N2-Depth-Play", "N2DepthEnvCfg_PLAY"),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{env_class}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:N2DepthPPORunnerCfg",
        },
    )
