# SPDX-License-Identifier: BSD-3-Clause
"""在 GPU 上检查 N2 导入、深度帧保持与重置、Actor 输入、PPO 更新及部署导出。"""

import argparse
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--output", type=str, default="/tmp/ame2_n2_check")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import ame2.tasks
from ame2.tasks.ame2_env_cfg import Ame2EnvCfg_G1
from ame2.tasks.agents.rsl_rl_ppo_cfg import PPORunnerCfg
from ame2.tasks.n2.agent_cfg import N2DepthPPORunnerCfg
from ame2.tasks.n2.depth import preprocess_depth
from ame2.tasks.n2.export import export_policy


def main():
    # 创建新配置时不应修改 G1 的类默认配置。
    cfg = parse_env_cfg("Ame2-N2-Depth-Play", device="cuda:0", num_envs=args.num_envs)
    cfg.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.0)
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 2
    cfg.scene.terrain.max_init_terrain_level = 0
    g1 = Ame2EnvCfg_G1()
    assert g1.scene.height_scanner is not None
    assert g1.scene.self_col_knee.prim_path.endswith("/left_knee_link")
    assert PPORunnerCfg().algorithm.symmetry_cfg is not None

    env = RslRlVecEnvWrapper(gym.make("Ame2-N2-Depth-Play", cfg=cfg))
    try:
        observations = env.get_observations()
        assert set(observations.keys()) == {"proprio", "commands", "depth", "critic"}
        assert observations["proprio"].shape == (args.num_envs, 300)
        assert observations["commands"].shape == (args.num_envs, 4)
        assert observations["depth"].shape == (args.num_envs, 1, 16, 24)
        assert env.num_actions == 18
        print("[N2 check] URDF、传感器、观测维度和 G1 配置隔离检查通过。", flush=True)

        camera = env.unwrapped.scene["depth_camera"]
        raw = camera.data.output["distance_to_image_plane"].clone()
        torch.testing.assert_close(observations["depth"], preprocess_depth(raw.permute(0, 3, 1, 2)))
        assert (observations["depth"] > -1.0).any(), "相机在策略有效深度范围内没有观测到地形。"
        previous_frame = camera.frame.clone()
        previous_depth = observations["depth"].clone()
        held_steps = 0
        for _ in range(12):
            observations, _, _, _ = env.step(torch.zeros(args.num_envs, 18, device=env.device))
            unchanged = camera.frame == previous_frame
            if unchanged.any():
                torch.testing.assert_close(observations["depth"][unchanged], previous_depth[unchanged])
                held_steps += 1
            previous_frame = camera.frame.clone()
            previous_depth = observations["depth"].clone()
        assert held_steps > 0
        print("[N2 check] 两次传感器更新之间正确保持了异步深度帧。", flush=True)

        agent_cfg = N2DepthPPORunnerCfg()
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.device)
        runner.learn(num_learning_iterations=1)
        policy = runner.alg.get_policy().eval()
        observations = env.get_observations()
        actor_only = observations.select("proprio", "commands", "depth")
        with torch.inference_mode():
            actions = policy(actor_only)
        assert torch.isfinite(actions).all()
        assert all(torch.isfinite(param).all() for param in policy.parameters())
        print("[N2 check] 实际仿真环境的一轮 PPO 更新通过；Actor 无需特权输入。", flush=True)

        # 采样结束后重新生成无噪声观测，用于比较部署包装器的输出。
        raw = camera.data.output["distance_to_image_plane"].permute(0, 3, 1, 2).clone()
        clean_obs = actor_only.clone()
        clean_obs["depth"] = preprocess_depth(raw)
        with torch.inference_mode():
            expected = policy(clean_obs).cpu()
        output = Path(args.output)
        exported = export_policy(policy, env, output)
        with torch.inference_mode():
            torch.testing.assert_close(policy(clean_obs).cpu(), expected)
        assert all(param.device == torch.device(env.device) for param in policy.parameters())
        deployed = torch.jit.load(str(exported))
        with torch.inference_mode():
            actual = deployed(clean_obs["proprio"].cpu(), clean_obs["commands"].cpu(), raw.cpu())
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        with torch.inference_mode():
            env.reset()
            after_reset = env.get_observations()
            torch.testing.assert_close(after_reset["depth"], preprocess_depth(
                camera.data.output["distance_to_image_plane"].permute(0, 3, 1, 2)))
        runner.save(str(output / "model_check.pt"))
        print(f"[N2 check] JIT 输出一致性和重置检查通过，导出位置：{exported}", flush=True)
    finally:
        env.close()


report = Path(args.output) / "check_result.json"
try:
    main()
except Exception:
    error = traceback.format_exc()
    print(error, flush=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"passed": False, "error": error}, indent=2) + "\n")
    raise
else:
    report.write_text(json.dumps({"passed": True}, indent=2) + "\n")
finally:
    app.close()
