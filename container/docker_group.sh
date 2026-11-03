#!/bin/bash
# 供容器脚本引用：用户已加入 docker 组，但当前终端尚未刷新权限时，自动重新执行。

ame2_refresh_docker_group() {
    local group_record group_name group_password group_id group_members
    local current_groups configured_groups command argument

    [ "${AME2_DOCKER_GROUP_REFRESHED:-0}" != 1 ] || return 0
    command -v getent >/dev/null 2>&1 || return 0
    command -v sg >/dev/null 2>&1 || return 0
    [ -S /var/run/docker.sock ] || return 0
    group_record=$(getent group docker) || return 0
    IFS=: read -r group_name group_password group_id group_members <<< "$group_record"
    [ "$(stat -c %g /var/run/docker.sock)" = "$group_id" ] || return 0

    current_groups=" $(id -G) "
    configured_groups=" $(id -G "$(id -un)") "
    [[ "$current_groups" != *" $group_id "* ]] || return 0
    [[ "$configured_groups" == *" $group_id "* ]] || return 0

    echo "[docker] 用户已加入 docker 组，正在为当前命令刷新组权限……" >&2
    # 使用 POSIX 单引号转义，完整保留空格、引号、换行和 Hydra 参数。
    command="exec /bin/bash"
    for argument in "$@"; do
        argument=${argument//\'/\'\\\'\'}
        command+=" '$argument'"
    done
    export AME2_DOCKER_GROUP_REFRESHED=1
    exec sg docker -c "$command"
}
