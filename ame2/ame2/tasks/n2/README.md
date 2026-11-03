# N2 深度输入一阶段 PPO 训练

新增任务为 `Ame2-N2-Depth` 和 `Ame2-N2-Depth-Play`。
原有 G1 的任务名、环境配置、网络及训练脚本保持原来的行为；
已有任务注册文件中只增加了 N2 注册模块的导入。

本任务复用 AME2 的目标到达奖励、地形课程、200 Hz 物理仿真、50 Hz 策略控制、
Muon 优化器、bfloat16 PPO 更新和多头 Critic。
Actor 使用仓库内置的 `CNNModel`：深度 CNN 提取视觉特征，再将视觉特征、
本体状态历史和目标指令输入 MLP。训练从深度图和本体状态直接开始，
不需要教师、LiDAR、神经建图权重、动作数据集或 AMP。
该 Actor 使用深度 CNN，不使用原有的 AME2 高度地图或 Gaze 编码器。

## 机器人配置

18 自由度 N2 的 URDF 和 21 个网格文件来自你提供的
`Noetix-Bumi/.../assets/robots/N2`，已复制到本仓库的 `ame2/ame2/assets/n2`。
资产目录保留了原始 BSD 许可证。初始姿态、PD 增益、力矩限制、速度限制和动作缩放
参考原项目的 `n2.py`。

机器人配置文件是 `ame2/ame2/assets/noetix.py`，使用 Isaac Lab 自带的隐式 PD 执行器。
动作延迟沿用 AME2 的动作模块：训练时为 0–1 个策略步，回放时为零。
运行时不需要导入另外两个参考项目。

`UrdfFileCfg` 在启动时将 URDF 转换为 USD，并保留固定连接的手部连杆。
转换结果缓存在容器内的 `/tmp/ame2/n2_usd`；因此每个新容器首次加载时会进行模型转换。

躯干、足端、膝部、髋部和手臂的选择配置已适配 N2 的实际连杆名称。
原 G1 镜像增强写死了 G1 的关节和接触顺序，因此 N2 训练配置关闭了该增强。
Critic 仍然接收训练用的真实状态和高度扫描，但 Actor 及部署导出不接收这些信息。

## 进入 Docker 环境

在宿主机的项目根目录执行：

```bash
cd /home/zsp/project/ame2_minimal
./container/run.sh -g 0 shell
```

进入后的目录为 `/workspace/ame2-minimal/ame2`。
容器内使用 `/isaac-sim/python.sh` 运行脚本，输入 `exit` 退出。
如果希望后台容器持续运行，可以在宿主机执行：

```bash
./container/run.sh -g 0 -d shell
./container/run.sh exec
```

启动脚本会自动处理已配置 Docker 用户组但当前终端尚未刷新权限的情况，直接使用上述命令即可。
也可以手动刷新当前命令的组权限：

```bash
sg docker -c './container/run.sh -g 0 shell'
```

其他环境操作见 [Docker 使用说明](../../../../container/README.md)。

## 相机与输入约定

相机标定和深度处理参考以下配置中的实际启用项：

- `Perceptive-locomotion/.../tasks/walkrun/noetix_n2/walkrun_amp_depth_cfg.py`
- 同目录的 `mdp/observations.py`

| 配置项 | 数值或含义 |
|---|---|
| 安装连杆 | `base_link` |
| 相对安装位置，单位 m | `(0.06387, 0.01629, -0.11057)` |
| ROS 坐标约定的四元数，顺序 wxyz | `(0.2705981, -0.6532815, 0.6532815, -0.2705981)` |
| 图像宽 × 高 | 24 × 16 |
| fx、fy、cx、cy | 12.6454、11.2403、12.0、8.0 |
| 深度含义 | 沿相机光轴的距离，`distance_to_image_plane`，单位 m |
| 射线最大距离 | 4 m |
| 策略使用的深度范围 | 0.3–2 m；无效或超范围像素置零，再整体减 1 |
| 深度更新周期 | 默认 0.1 s，与 0.02 s 的策略周期独立 |

参考配置的相机射线距离为 4 m，但其**实际启用的观测处理显式传入了 2 m 上限**，
因此本任务训练和部署时都按 2 m 处理。参考环境还会在初始化时将相机更新周期覆盖为
0.015 s；根据你的要求，本任务采用 AME2 感知侧的 10 Hz 更新节奏，独立于策略周期。
两次相机更新之间，策略使用最近一帧深度图，对应的噪声也随该帧保持不变。

当前相机通过 Warp 对静态地形网格进行射线计算，不需要 RTX 相机渲染，
也不需要传入 `--enable_cameras`。当前没有模拟动态机器人身体对相机的遮挡。

Actor 接收以下观测，其中 `B` 表示并行环境数量：

- `proprio [B,300]`：5 帧本体状态，按从旧到新的顺序展开。每帧 60 维，
  包含机身坐标系角速度 3 维、重力投影 3 维、相对默认姿态的关节位置 18 维、
  关节速度 18 维，以及上一条已施加的策略动作 18 维。
  重置时使用首帧填充历史，不输入机身真实线速度。
- `commands [B,4]`：`[goal_x_body, goal_y_body, sin(heading_error), cos(heading_error)]`。
  沿用 AME2 的目标位置任务，目标位置向量长度限制为 2 m。
  该指令和另一个 N2 项目的三维速度指令不同。
- `depth [B,1,16,24]`：最近一帧经过预处理的深度图。

训练时加入 AME2 风格的本体状态噪声，以及随距离变化的深度噪声、高斯噪声和椒盐噪声；
回放时关闭这些噪声。当前没有引入参考项目完整的持续高反、条纹和矩形遮挡噪声流程。

## 启动训练

镜像已经构建时可以直接训练。如果需要首次配置或重新构建，在宿主机项目根目录执行：

```bash
./container/setup.sh
```

在宿主机直接启动 N2 训练：

```bash
./container/run.sh -g 0 train --task Ame2-N2-Depth --num_envs 1024
```

也可以先进入容器，再在容器内执行：

```bash
/isaac-sim/python.sh scripts/rsl_rl/train.py --headless \
  --task Ame2-N2-Depth --num_envs 1024
```

源码已经挂载到容器，新增或修改本任务后无需重建镜像。
可通过 Hydra 单独修改相机周期，例如在宿主机执行：

```bash
./container/run.sh -g 0 train --task Ame2-N2-Depth --num_envs 1024 \
  env.scene.depth_camera.update_period=0.0666667
```

训练日志和 checkpoint 位于宿主机的 `ame2/logs/rsl_rl/ame2_n2_depth/`，
在容器内对应 `logs/rsl_rl/ame2_n2_depth/`。
需要新建训练运行；G1 权重和另一个 N2 项目的 AMP 权重与当前 Actor 不兼容。

## 验证

进入容器后执行：

```bash
/isaac-sim/python.sh -m unittest discover -s scripts/rsl_rl -p test_n2_depth.py -v
/isaac-sim/python.sh scripts/rsl_rl/check_n2_depth.py --headless
```

第一条命令在 CPU 上检查深度范围、无效值处理和 TorchScript 一致性。
第二条命令默认使用两个仿真环境，检查 URDF 加载、传感器、观测维度、
G1 配置隔离、异步深度帧保持与重置、一轮 PPO 更新，以及部署导出的一致性。
这些是功能检查，不代表已经获得稳定步态或完成实机迁移。

## 回放与部署导出

原有 G1 回放脚本包含教师地图专用的可视化和导出输入，N2 使用独立回放脚本。
需要图形界面时，先在宿主机执行：

```bash
./container/run.sh -g 0 --gui shell
```

再在容器内执行，替换实际训练目录和 checkpoint：

```bash
/isaac-sim/python.sh scripts/rsl_rl/play_n2_depth.py \
  --checkpoint logs/rsl_rl/ame2_n2_depth/<run>/model_4999.pt --export
```

无界面回放时，使用普通 `./container/run.sh shell` 进入容器，并在上述 Python 命令后
添加 `--headless --steps 100`，运行 100 个策略步后结束。
`--camera_period 0.1` 可单独设置回放时的相机周期。

导出文件位于 checkpoint 同目录下的 `exported/`：

- `n2_depth_policy.jit`：部署用 TorchScript 策略。
- `n2_depth_policy.json`：实际仿真关节顺序、动作顺序、PD 增益、限制和相机标定信息。

JIT 接口为：

```text
输入：proprio [B,300]、commands [B,4]、depth_m [B,1,16,24]
输出：原始动作 actions [B,18]
```

`depth_m` 必须是以米为单位的光轴深度。JIT 已包含有效范围过滤、深度减 1 的处理，
以及本体状态归一化。部署端按照 JSON 中的实际关节和动作顺序应用动作缩放及默认关节偏置，
并使用相应 PD 增益和关节限制。
部署端还需要维护本体状态历史、最近一帧异步深度数据，并从 IMU 或状态估计器获得重力投影。
