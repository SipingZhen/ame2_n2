# SPDX-License-Identifier: BSD-3-Clause
"""基于 AME2 目标到达任务的 N2 深度相机配置。

适配仅作用于本任务自己的配置实例。Actor 不接收高度扫描、
接触状态、真实线速度或教师观测。
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import RayCasterCameraCfg, patterns
from isaaclab.utils import configclass

from ame2.assets.noetix import N2_ACTION_SCALE, N2_BASE_LINK, N2_CFG, N2_FEET, N2_HIPS, N2_MASS_KG, N2_STANDING_HEIGHT

from .. import mdp
from ..ame2_env_cfg import Ame2EnvCfg_G1, Ame2EnvCfg_G1_PLAY, ObservationsCfg
from .depth import (
    CAMERA_INTRINSICS,
    CAMERA_POSITION,
    CAMERA_ROTATION,
    DEPTH_HEIGHT,
    DEPTH_RAY_RANGE,
    DEPTH_UPDATE_PERIOD,
    DEPTH_WIDTH,
    PROPRIO_HISTORY,
)
from .observations import DepthImage, proprioception


@configclass
class ProprioObsCfg(ObservationGroupCfg):
    proprio = ObservationTermCfg(
        func=proprioception,
        params={"noisy": True},
        history_length=PROPRIO_HISTORY,
        flatten_history_dim=True,
    )

    def __post_init__(self):
        self.enable_corruption = False  # 各分量的噪声由 proprioception() 单独处理。
        self.concatenate_terms = True


@configclass
class DepthObsCfg(ObservationGroupCfg):
    image = ObservationTermCfg(
        func=DepthImage,
        params={"sensor_cfg": SceneEntityCfg("depth_camera"), "noisy": True},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


def _configure_n2_depth(cfg, play: bool = False):
    cfg.scene.robot = N2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.actions.joint_pos.joint_names = ["[LR]_leg_.*_joint", "[LR]_arm_.*_joint"]
    cfg.actions.joint_pos.scale = N2_ACTION_SCALE.copy()

    # 特权地形信息仅供 Critic 和奖励计算使用。
    cfg.scene.height_scanner = None
    cfg.scene.height_scanner_critic.prim_path = f"{{ENV_REGEX_NS}}/Robot/{N2_BASE_LINK}"
    cfg.scene.height_scanner_critic.update_period = cfg.decimation * cfg.sim.dt
    cfg.observations.teacher_mapping = None
    cfg.observations.teacher_prop = None
    cfg.curriculum.teacher_mapping_noise = None
    cfg.observations.critic.height_scan.params["offset"] = N2_STANDING_HEIGHT
    cfg.observations.critic.contact_states.params["ignore_bodies"] = []

    # 标定参考 Perceptive-locomotion/noetix_n2；采用 AME2 的 10 Hz 感知节奏，
    # 与 50 Hz 策略独立。Warp 射线计算无需 RTX 相机渲染。
    cfg.scene.depth_camera = RayCasterCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{N2_BASE_LINK}",
        mesh_prim_paths=["/World/ground"],
        offset=RayCasterCameraCfg.OffsetCfg(pos=CAMERA_POSITION, rot=CAMERA_ROTATION, convention="ros"),
        pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
            intrinsic_matrix=list(CAMERA_INTRINSICS), width=DEPTH_WIDTH, height=DEPTH_HEIGHT, focal_length=24.0,
        ),
        data_types=["distance_to_image_plane"],
        update_period=DEPTH_UPDATE_PERIOD,
        max_distance=DEPTH_RAY_RANGE,
        depth_clipping_behavior="zero",
        debug_vis=False,
    )
    cfg.observations.proprio = ProprioObsCfg()
    cfg.observations.commands = ObservationsCfg.StudentCmdsObsCfg()
    cfg.observations.depth = DepthObsCfg()

    # 保留自接触奖励项，按 N2 实际连杆名称及大小写重新配置。
    cfg.scene.self_col_knee.prim_path = "{ENV_REGEX_NS}/Robot/L_leg_knee_link"
    cfg.scene.self_col_knee.filter_prim_paths_expr = ["{ENV_REGEX_NS}/Robot/R_leg_knee_link"]
    cfg.scene.self_col_ankle.prim_path = "{ENV_REGEX_NS}/Robot/L_leg_ankle_link"
    cfg.scene.self_col_ankle.filter_prim_paths_expr = ["{ENV_REGEX_NS}/Robot/R_leg_ankle_link"]
    cfg.scene.self_col_torso.prim_path = f"{{ENV_REGEX_NS}}/Robot/{N2_BASE_LINK}"
    cfg.scene.self_col_torso.filter_prim_paths_expr = [
        f"{{ENV_REGEX_NS}}/Robot/{side}_arm_{part}_Link"
        for side in ("L", "R") for part in ("shoulder_roll", "shoulder_yaw", "elbow")
    ]

    # 将原先选择 G1 躯干或头部的质量、质心随机化和奖励项适配到 N2。
    for event_name in ("add_base_mass", "base_com", "assistive_force"):
        term = getattr(cfg.events, event_name)
        if term is not None:
            term.params["asset_cfg"] = SceneEntityCfg("robot", body_names=N2_BASE_LINK)
    for reward_name in ("ame2_move2goal", "undesired_spinning", "base_roll_rate"):
        getattr(cfg.rewards, reward_name).params["asset_cfg"] = SceneEntityCfg("robot", body_names=N2_BASE_LINK)

    feet = SceneEntityCfg("contact_forces", body_names=N2_FEET, preserve_order=True)
    cfg.rewards.ame2_standatgoal.params["sensor_cfg"] = feet
    cfg.rewards.undesired_leaping.params["feet_sensor_cfg"] = feet
    cfg.rewards.undesired_non_foot_contacts.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names="^(?![LR]_leg_ankle_link$).*",
    )
    for reward_name in ("undesired_stumbling", "undesired_slippage", "link_contact_forces"):
        getattr(cfg.rewards, reward_name).params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=".*")
    cfg.rewards.link_contact_forces.params["robot_weight"] = N2_MASS_KG * 9.81
    cfg.rewards.body_lin_acc_l1.params["asset_cfg"] = SceneEntityCfg("robot", body_names=".*")
    cfg.rewards.ame2_optional_uppershaping.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names="[LR]_arm_.*_joint",
    )
    cfg.rewards.ame2_optional_foot_shaping.params["asset_cfg"] = SceneEntityCfg("robot", body_names=N2_FEET)
    cfg.rewards.ame2_optional_foot_shaping.params["torso_cfg"] = SceneEntityCfg("robot", body_names=N2_BASE_LINK)

    cfg.terminations.early_gravity.params["asset_cfg"] = SceneEntityCfg("robot", body_names=N2_BASE_LINK)
    cfg.terminations.early_hip_acc.params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=N2_HIPS, preserve_order=True,
    )
    cfg.terminations.early_hip_acc.params["foot_sensor_cfg"] = feet
    cfg.terminations.early_illegal_contact.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=N2_BASE_LINK,
    )

    # 使用本地生成的外观，避免加载 G1 的远程 HDR 和材质资产。
    cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4))
    cfg.scene.sky_light.spawn = sim_utils.DomeLightCfg(intensity=750.0)
    cfg.scene.terrain.max_init_terrain_level = 1
    if play:
        cfg.observations.proprio.proprio.params["noisy"] = False
        cfg.observations.depth.image.params["noisy"] = False


@configclass
class N2DepthEnvCfg(Ame2EnvCfg_G1):
    def __post_init__(self):
        super().__post_init__()
        _configure_n2_depth(self)


@configclass
class N2DepthEnvCfg_PLAY(Ame2EnvCfg_G1_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _configure_n2_depth(self, play=True)
