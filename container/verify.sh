#!/bin/bash
# 使用与训练相同的源码挂载和 GPU 选项验证已构建的镜像。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -ne 0 ]; then
    echo "用法：GPU_ID=0 IMAGE=ame2-minimal:latest ./container/verify.sh" >&2
    exit 2
fi

"$HERE/run.sh" -g "${GPU_ID:-0}" shell <<'CONTAINER'
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
/isaac-sim/python.sh - <<'PY'
from importlib import metadata
from pathlib import Path
import math

import torch

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch 无法访问 CUDA，请检查 NVIDIA Container Toolkit 和 Docker GPU 透传。")
result = torch.ones(32, device="cuda:0").sum().item()
assert result == 32, result
print(f"[verify] CUDA 计算通过：{torch.cuda.get_device_name(0)}", flush=True)

for package in ("torch", "tensordict", "rsl-rl-lib", "ame2"):
    print(f"[verify] {package}: {metadata.version(package)}", flush=True)

from isaaclab.app import AppLauncher

launcher = AppLauncher(headless=True)
app = launcher.app
try:
    import gymnasium as gym
    import isaaclab_tasks
    import rsl_rl
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import CNNModel, MLPModel
    from rsl_rl.storage import RolloutStorage
    from tensordict import TensorDict
    from isaaclab_tasks.utils import parse_env_cfg
    import ame2.tasks

    expected_rsl_path = Path("/workspace/ame2-minimal/rsl_rl/rsl_rl/__init__.py")
    if Path(rsl_rl.__file__).resolve() != expected_rsl_path:
        raise RuntimeError(f"加载了错误的 rsl_rl：{rsl_rl.__file__}；应加载 AME2 仓库内置实现。")

    # 使用模拟观测检查 CNN、PPO 和 CUDA，无需机器人资产或物理仿真环境。
    obs = TensorDict({
        "policy": torch.randn(4, 10, device="cuda:0"),
        "depth": torch.rand(4, 1, 16, 24, device="cuda:0"),
        "critic": torch.randn(4, 12, device="cuda:0"),
    }, batch_size=[4], device="cuda:0")
    groups = {"actor": ["policy", "depth"], "critic": ["critic"]}
    actor = CNNModel(obs, groups, "actor", 4, hidden_dims=[32],
                     cnn_cfg={"output_channels": [8], "kernel_size": 3, "stride": 2},
                     distribution_cfg={"class_name": "GaussianDistribution", "std_type": "log"})
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[32])
    storage = RolloutStorage("rl", 4, 4, obs, [4], device="cuda:0")
    ppo = PPO(actor, critic, storage, device="cuda:0", multi_head_critic=False,
              optimizer="muon", use_mixed_precision=True, num_learning_epochs=1, num_mini_batches=1)
    with torch.inference_mode():
        for _ in range(4):
            ppo.act(obs)
            ppo.process_env_step(obs, torch.randn(4, device="cuda:0"),
                                 torch.zeros(4, dtype=torch.bool, device="cuda:0"), {})
        ppo.compute_returns(obs)
    losses = ppo.update()
    if not all(math.isfinite(value) for value in losses.values()):
        raise RuntimeError(f"PPO 损失出现非有限值：{losses}")
    print("[verify] 模拟深度观测的 CNN PPO 更新通过，包含 Muon 和 bfloat16。", flush=True)

    tasks = sorted(name for name in gym.registry if name.startswith("Ame2-"))
    for task in tasks:
        print(f"[verify] 已注册任务：{task}", flush=True)
    if "Ame2-G1-Gaze" not in tasks:
        raise RuntimeError("AME2 任务注册失败。")

    cfg = parse_env_cfg("Ame2-G1-Gaze", device="cuda:0", num_envs=1)
    assert cfg.scene.num_envs == 1
    print("[verify] Isaac Sim 启动、AME2 导入和环境配置检查通过。", flush=True)
finally:
    app.close()
PY
/isaac-sim/python.sh scripts/rsl_rl/train.py --help
echo "[verify] 容器环境检查通过，本次未加载机器人资产或进行机器人训练。"
CONTAINER
