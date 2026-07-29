# SimpleCNN 流式推理

统一入口：

    cd /mnt/host-model/weixc/code/LineTracker/method/SimpleCNN

    python -m infer.run_infer \
      --method global_top1 \
      --checkpoint /绝对路径/训练run/checkpoints/best.pt \
      --num-samples 1 \
      --device npu:0 \
      --time-stride 5

global_top1 每个流式窗口把全部 34 个标准距离块组成一个 batch，仅按 q 取 Top-1，再从前 20 帧外推未来 time_stride 帧。

严格按 _doc/postprocess.md 的自适应方法：

    python -m infer.run_infer \
      --method adaptive_tracker \
      --checkpoint /绝对路径/训练run/checkpoints/best.pt \
      --num-samples 10 \
      --device npu:0 \
      --time-stride 5 \
      --capture-stride 3 \
      --capture-buffer-size 8 \
      --capture-support-ratio 0.7 \
      --capture-radius-m 500 \
      --q-keep 0.5 \
      --position-gates-m 1000,2000,4000

入口读取 checkpoint 同一 run 的 resolved_config.json 以恢复模型和数据参数；样本直接从当前 data_root 的一级序列目录按返回顺序取前 N 条，不读取训练 run 的 split manifest。跨机器时可用 --data-root /当前机器/数据根目录 覆盖默认数据位置。

同一数据集与 checkpoint 的结果稳定写入同一个目录；先运行一种方法、后运行另一种方法时，会在相同样本目录内追加对应文件：

    eval/_output/<数据集简称>--<ckpt简称>/
    ├── infer_config.json
    └── samples/<source_id>/
        ├── global_top1_stride5/
        │   ├── metrics.json
        │   ├── log.jsonl
        │   ├── visualize.png
        │   └── prediction.npz
        └── adaptive_tracker_stride5/
            ├── metrics.json
            ├── log.jsonl
            ├── visualize.png
            └── prediction.npz

每个 `<method>_strideN/metrics.json` 是带缩进的可读汇总，`log.jsonl` 是逐窗推理日志，`visualize.png` 是中文 2×2 诊断图；真值和 target_hit 只用于离线指标/诊断，不会参与块选择或状态机。--no-figures 可只保存数值结果；--no-warmup 可跳过不计时的 1/3/5/34 块预热。单次入口只使用一张由 --device 指定的 CUDA/NPU 卡。

