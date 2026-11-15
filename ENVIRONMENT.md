# 运行环境

本机环境：`/SharePath/weixc/.conda/linetracker-py311`，Python 3.11、PyTorch 2.7.1、torch_npu 2.7.1.post4。

在仓库根目录激活：

```bash
source activate_env.sh
```

脚本会读取 `method/SimpleCNN/.env`，激活 Conda 并设置运行时变量。单独执行 `conda activate` 不会加载项目配置。

- `.env`：Conda、数据/输出路径、驱动库路径、W&B 地址及共用运行参数。换机器时修改这里，不要提交 API Key。
- `.vscode/launch.json`：各启动项分别设置 CUDA/NPU 卡号。
- VS Code：通过 `Python: Select Interpreter` 选择上述环境的 `bin/python`。

Ascend 运行库路径不要加入 `devlib`；不要用通用 PyTorch 包覆盖厂商版本。环境已安装中文字体。

纯 CPU 运行可在命令前加 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，避免自动加载 NPU 扩展。`conda deactivate` 不会自动清除脚本导出的项目变量。
