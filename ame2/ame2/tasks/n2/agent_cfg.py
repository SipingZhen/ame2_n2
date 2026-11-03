# SPDX-License-Identifier: BSD-3-Clause
"""一阶段非对称 PPO：Actor 使用深度和本体状态，Critic 使用特权信息。"""

from isaaclab.utils import configclass

from ..agents.rsl_rl_ppo_cfg import PPORunnerCfg


@configclass
class N2DepthActorCfg:
    class_name: str = "rsl_rl.models.cnn_model:CNNModel"
    hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    obs_normalization: bool = True
    cnn_cfg: dict = {
        "output_channels": [16, 32],
        "kernel_size": 3,
        "stride": 2,
        "padding": "zeros",
        "activation": "elu",
        "flatten": True,
    }
    distribution_cfg: dict = {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "log",
    }


@configclass
class N2DepthPPORunnerCfg(PPORunnerCfg):
    experiment_name = "ame2_n2_depth"
    obs_groups = {"actor": ["proprio", "commands", "depth"], "critic": ["critic"]}
    actor = N2DepthActorCfg()

    def __post_init__(self):
        # 原镜像增强函数写死了 G1 的 23 个关节和 26 个接触状态，因此关闭。
        self.algorithm.symmetry_cfg = None
