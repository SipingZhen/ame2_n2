# AME2 Docker 环境使用说明

## 进入容器

在宿主机终端切换到项目根目录，启动一个交互式容器：

```bash
cd /home/zsp/project/ame2_minimal
./container/run.sh -g 0 shell
```

`-g 0` 表示使用第 0 块 GPU。进入后，当前目录是容器内的
`/workspace/ame2-minimal/ame2`，运行 Python 脚本时使用 Isaac Sim 自带的解释器：

```bash
/isaac-sim/python.sh scripts/rsl_rl/train.py --headless \
  --task Ame2-N2-Depth --num_envs 1024
```

输入 `exit` 退出。上述前台容器退出后会自动删除；挂载的项目代码和训练日志仍保留在宿主机。

如果希望容器一直在后台运行，可以启动后台容器，再进入：

```bash
./container/run.sh -g 0 -d shell
./container/run.sh exec
```

`exec` 默认进入最近启动的 `ame2_*` 容器。也可以指定启动时输出的容器名：

```bash
./container/run.sh exec <容器名>
```

退出 `exec` 打开的终端不会关闭后台容器。查看和停止后台容器：

```bash
docker ps --filter 'name=ame2_'
docker stop <容器名>
```

需要图形界面时，在宿主机已配置好 X11 显示转发的前提下使用：

```bash
./container/run.sh -g 0 --gui shell
```

启动、构建和配置脚本会自动检测“用户已加入 `docker` 组，但当前终端尚未刷新权限”的情况，
并通过 `sg docker` 重新执行当前命令，无需手动加 `sudo` 或修改 socket 权限。
如果需要让当前终端直接执行其他 Docker 命令，可新开终端或执行 `newgrp docker`。
也可以手动只为当前命令刷新用户组：

```bash
sg docker -c './container/run.sh -g 0 shell'
```

## 构建与验证

环境配置以 `nvcr.io/nvidia/isaac-lab:2.3.2` 为基础镜像，包含 Isaac Sim 5.1，
项目镜像名默认为 `ame2-minimal:latest`。首次配置或需要重建时，在项目根目录执行：

```bash
./container/setup.sh --check-only
./container/setup.sh
```

`setup.sh` 首先检查 Docker 服务访问权限、宿主机 NVIDIA 驱动和 Buildx；
然后调用已有的 `build.sh` 构建镜像，并运行 `verify.sh` 验证环境。
首次构建需要联网下载 NVIDIA 基础镜像，并预留相应磁盘空间。

验证脚本沿用 `run.sh` 的 GPU 选择和源码挂载方式，检查以下内容：

- 容器内的 NVIDIA GPU 访问，以及一次 CUDA 张量计算。
- 软件包版本，以及是否加载了本仓库内置的 RSL-RL。
- Isaac Sim 无界面启动。
- 使用模拟深度观测执行一轮 CNN PPO 更新，包含 Muon 和 bfloat16。
- AME2 任务注册、环境配置及训练命令行入口。

该环境验证不会创建机器人仿真实例，也不会进行机器人训练。

单独验证已有镜像，或指定其他 GPU：

```bash
GPU_ID=1 ./container/verify.sh
IMAGE=ame2-minimal:latest GPU_ID=0 ./container/setup.sh
```

## 直接启动训练

无需先进入容器，也可以直接在宿主机启动训练：

```bash
# 新增的 N2 深度输入任务
./container/run.sh -g 0 train --task Ame2-N2-Depth --num_envs 1024

# 原有的 G1 Gaze 教师任务
./container/run.sh -g 0 train --task Ame2-G1-Gaze --num_envs 4096
```

`run.sh` 将宿主机的 `ame2/` 和 `rsl_rl/` 挂载到容器中，修改源码后无需重建镜像。
训练日志保存在宿主机的 `ame2/logs/`，在容器内对应当前工作目录下的 `logs/`。
N2 的观测、相机、回放和部署说明见 [N2 任务说明](../ame2/ame2/tasks/n2/README.md)。

## 宿主机前置条件

如果 Docker 和 NVIDIA 容器运行时已经可用，直接使用即可。
配置脚本不会修改 Docker 服务配置、修改用户组或重启服务。

- Docker Engine 和 Buildx 的安装参考 [Docker 官方文档](https://docs.docker.com/engine/install/)。
- 先确认宿主机执行 `nvidia-smi` 正常。受限终端看不到 GPU，并不意味着宿主机驱动损坏。
- NVIDIA Container Toolkit 的安装和配置参考 [NVIDIA 官方文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。

如果 Docker 服务已经启动，但宿主机用户尚无访问权限，参考
[Docker 的 Linux 安装后配置](https://docs.docker.com/engine/install/linux-postinstall/)。
普通 Linux 宿主机使用 `docker` 用户组时，可以执行：

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

仅在尚未配置 NVIDIA 容器运行时时，才需要在宿主机执行以下命令。
重启 Docker 可能影响正在运行的容器，应选择合适的时间操作：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

受限终端可能看得见 Docker 命令和 socket，却无法连接 Docker、使用 `sudo` 或访问
`/dev/nvidia*`。遇到这种情况，应在有对应权限的宿主机终端运行配置和验证。

## G1 机器人资产

新克隆仓库中的 G1 USD 文件可能仍是 Git LFS 指针。在运行 G1 仿真之前，
需要在宿主机下载真实资产：

```bash
git lfs install
git lfs pull
```

容器环境验证不会自动下载这些资产。原有教师、学生和 Gaze 实现保留。
