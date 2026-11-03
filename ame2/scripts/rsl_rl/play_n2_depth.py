# SPDX-License-Identifier: BSD-3-Clause
"""回放和导出独立的 N2 深度策略，无需 G1 专用可视化。"""

import argparse
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=0, help="运行指定数量的策略步后结束；零表示持续运行到关闭。")
parser.add_argument("--export", action="store_true", help="保存部署 JIT 和机器人、相机元数据。")
parser.add_argument("--camera_period", type=float, default=None, help="独立的深度相机更新周期，单位秒。")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch
import yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import ame2.tasks
from ame2.tasks.n2.agent_cfg import N2DepthPPORunnerCfg
from ame2.tasks.n2.export import export_policy


def main():
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg = parse_env_cfg("Ame2-N2-Depth-Play", device=args.device or "cuda:0", num_envs=args.num_envs)
    if args.camera_period is not None:
        if args.camera_period <= 0:
            raise ValueError("--camera_period 必须大于零。")
        cfg.scene.depth_camera.update_period = args.camera_period

    saved_cfg = checkpoint.parent / "params" / "agent.yaml"
    if saved_cfg.is_file():
        agent_cfg = yaml.safe_load(saved_cfg.read_text())
    else:
        agent_cfg = N2DepthPPORunnerCfg().to_dict()
    env = RslRlVecEnvWrapper(gym.make("Ame2-N2-Depth-Play", cfg=cfg))
    try:
        runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=env.device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, map_location=env.device)
        policy = runner.get_inference_policy()
        if args.export:
            path = export_policy(runner.alg.get_policy(), env, checkpoint.parent / "exported")
            print(f"[N2] 部署策略已导出：{path}", flush=True)
        observations = env.get_observations()
        step = 0
        while app.is_running() and (args.steps == 0 or step < args.steps):
            with torch.inference_mode():
                actions = policy(observations)
                observations, _, dones, _ = env.step(actions)
                policy.reset(dones)
            step += 1
        print(f"[N2] 已完成 {step} 个策略步。", flush=True)
    finally:
        env.close()


try:
    main()
except Exception:
    print(traceback.format_exc(), flush=True)
    raise
finally:
    app.close()
