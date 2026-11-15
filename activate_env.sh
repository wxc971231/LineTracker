#!/usr/bin/env bash
# 用法：source /path/to/LineTracker/activate_env.sh
# 必须在 Bash 中 source，才能修改当前终端的环境；不要用 bash activate_env.sh 代替。
# 脚本不写死本机路径；换机器时需修改 method/SimpleCNN/.env 中的 Conda、数据和库路径。
_lt_activate_project_env() {
    # 使用函数和局部变量，避免临时变量留在调用者的终端中。
    local env_file root old_allexport
    # 按脚本自身的位置定位仓库，不依赖执行 source 时所在的工作目录。
    root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    env_file="$root/method/SimpleCNN/.env"
    if [[ ! -f "$env_file" ]]; then
        echo "Missing local environment file: $env_file" >&2
        return 1
    fi
    # 暂时开启自动导出，让 .env 的变量能被后续 Python 子进程继承。
    # 同时记住终端原来的设置，读取后恢复，避免改变用户的 Shell 行为。
    old_allexport=false
    [[ $- == *a* ]] && old_allexport=true
    set -a
    source "$env_file"
    [[ "$old_allexport" == true ]] || set +a
    # 这两个路径来自本机 .env：一个用于初始化 conda 命令，一个指定目标环境。
    if [[ -z "${LT_CONDA_SH-}" || -z "${LT_CONDA_PREFIX-}" ]]; then
        echo "Set LT_CONDA_SH and LT_CONDA_PREFIX in $env_file" >&2
        return 1
    fi
    # 初始化 Conda 并激活指定环境；任一步失败就停止，避免继续使用错误的 Python。
    source "$LT_CONDA_SH" || return
    conda activate "$LT_CONDA_PREFIX" || return
    # Conda 激活钩子可能修改库路径等变量，所以再次加载 .env，以项目配置为准。
    # 例如确保 LD_LIBRARY_PATH 优先使用真实驱动库，避免误加载 devlib 桩库。
    set -a
    source "$env_file"
    [[ "$old_allexport" == true ]] || set +a
}
# 执行后删除临时函数；失败时把非零状态返回给调用者，不退出整个终端。
if _lt_activate_project_env; then
    unset -f _lt_activate_project_env
else
    unset -f _lt_activate_project_env
    return 1
fi
