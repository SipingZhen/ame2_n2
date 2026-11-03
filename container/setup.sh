#!/bin/bash
# 在 GPU 宿主机运行：检查前置条件、构建 AME2 镜像并验证容器。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" = "--help" ]; then
    echo "用法：./container/setup.sh [--check-only]"
    echo "环境变量：GPU_ID=0 IMAGE=ame2-minimal:latest BASE_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2"
    exit 0
fi
if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != "--check-only" ]; }; then
    echo "用法：./container/setup.sh [--check-only]" >&2
    exit 2
fi

source "$HERE/docker_group.sh"
ame2_refresh_docker_group "$HERE/setup.sh" "$@"

for executable in docker nvidia-smi; do
    if ! command -v "$executable" >/dev/null 2>&1; then
        echo "[setup] 缺少 $executable，宿主机前置条件请参阅 container/README.md。" >&2
        exit 1
    fi
done

echo "[setup] 正在检查 Docker 服务访问权限……"
if ! docker info >/dev/null; then
    echo "[setup] 当前终端无法访问 Docker 服务。" >&2
    echo "[setup] 请在有 Docker socket 访问权限的 GPU 宿主机终端运行，参阅 container/README.md。" >&2
    exit 1
fi

echo "[setup] 正在检查宿主机 NVIDIA 驱动……"
if ! nvidia-smi -L; then
    echo "[setup] 当前终端无法访问正常工作的 NVIDIA 驱动。" >&2
    echo "[setup] 请先在 GPU 宿主机检查驱动，参阅 container/README.md。" >&2
    exit 1
fi

echo "[setup] 正在检查 Docker Buildx……"
if ! docker buildx version; then
    echo "[setup] 请先安装 Docker Buildx 插件，再运行 container/build.sh。" >&2
    exit 1
fi

if [ "${1:-}" = "--check-only" ]; then
    echo "[setup] 宿主机检查通过，镜像构建后将继续验证 GPU 透传。"
    exit 0
fi

"$HERE/build.sh"
"$HERE/verify.sh"
