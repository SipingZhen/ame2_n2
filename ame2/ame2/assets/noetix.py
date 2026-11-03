# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""N2 18 自由度机器人资产，参考 Noetix-Bumi 的 robots/N2/n2.py。

使用 Isaac Lab 的隐式 PD 执行器，并沿用原始增益和限制。
动作延迟由 AME2 动作模块提供，无需依赖外部执行器包。
"""

import os
import tempfile

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from ame2.assets import ISAAC_ASSET_DIR

N2_BASE_LINK = "base_link"
N2_FEET = ["L_leg_ankle_link", "R_leg_ankle_link"]
N2_HIPS = ["L_leg_hip_pitch_link", "R_leg_hip_pitch_link"]
N2_STANDING_HEIGHT = 0.75
N2_MASS_KG = 33.16351  # N2.urdf 中各连杆惯性质量的总和。

N2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAAC_ASSET_DIR}/n2/urdf/N2.urdf",
        usd_dir=os.path.join(tempfile.gettempdir(), "ame2", "n2_usd"),
        usd_file_name="n2.usd",
        fix_base=False,
        merge_fixed_joints=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, N2_STANDING_HEIGHT),
        joint_pos={
            "[LR]_leg_hip_yaw_joint": 0.0,
            "[LR]_leg_hip_roll_joint": 0.0,
            "[LR]_leg_hip_pitch_joint": -0.1495,
            "[LR]_leg_knee_joint": 0.3215,
            "[LR]_leg_ankle_joint": -0.1720,
            "[LR]_arm_shoulder_pitch_joint": 0.0,
            "L_arm_shoulder_roll_joint": 0.2,
            "R_arm_shoulder_roll_joint": -0.2,
            "[LR]_arm_shoulder_yaw_joint": 0.0,
            "[LR]_arm_elbow_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_leg_hip_.*_joint", "[LR]_leg_knee_joint"],
            effort_limit_sim={
                "[LR]_leg_hip_(yaw|roll)_joint": 90.0,
                "[LR]_leg_hip_pitch_joint": 150.0,
                "[LR]_leg_knee_joint": 150.0,
            },
            velocity_limit_sim=14.0,
            stiffness={
                "[LR]_leg_hip_(yaw|roll)_joint": 80.0,
                "[LR]_leg_hip_pitch_joint": 120.0,
                "[LR]_leg_knee_joint": 120.0,
            },
            damping=5.0,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_leg_ankle_joint"],
            effort_limit_sim=70.0,
            velocity_limit_sim=14.0,
            stiffness=20.0,
            damping=2.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_arm_.*_joint"],
            effort_limit_sim=27.0,
            velocity_limit_sim=10.0,
            stiffness=30.0,
            damping=1.0,
        ),
    },
)

# 沿用原始 N2 动作缩放：0.25 × 最大力矩 / PD 刚度。
N2_ACTION_SCALE = {
    "[LR]_leg_hip_(yaw|roll)_joint": 0.25 * 90.0 / 80.0,
    "[LR]_leg_hip_pitch_joint": 0.25 * 150.0 / 120.0,
    "[LR]_leg_knee_joint": 0.25 * 150.0 / 120.0,
    "[LR]_leg_ankle_joint": 0.25 * 70.0 / 20.0,
    "[LR]_arm_.*_joint": 0.25 * 27.0 / 30.0,
}
