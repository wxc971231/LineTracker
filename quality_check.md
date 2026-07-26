整体结论：代码主流程可以运行，静态编译也通过，但目前不建议直接扩大到大数据量长期多卡训练。存在 2 个明确的正确性问题，以及几处会让 validation/cache 在大规模数据下明显变慢的设计问题。

本次仅质检，没有修改代码。

## 高优先级问题

1. 独立评估入口已经失效

[evaluate.py](/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/eval/evaluate.py:49) 仍然从：

```text
run_dir/data/validation_grid.npz
run_dir/data/test_grid.npz
```

读取数据，但当前实现已经把网格放进全局 `_cache/val_test_grid`。实际运行目录中只有 `split_manifest.json`，按 README 中的独立评估命令会触发 `FileNotFoundError`。

建议让独立评估也调用 `prepare_data_artifacts()`，使用其返回的 `validation_manifest_path` 或 `test_manifest_path`。

2. 多卡下 BatchNorm 状态不一致

[model.py](/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/train/model.py:19) 使用了多层 `BatchNorm2d`，但 [trainer.py](/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/train/trainer.py:49) 设置了：

```python
broadcast_buffers=False
```

DDP 只同步参数梯度，不会同步各卡上的 `running_mean` 和 `running_var`。结果是：

- 不同 rank 实际使用不同的 BN 统计量做 validation；
- 汇总的验证指标来自多个略有差异的模型；
- 最终 checkpoint 只保存 rank 0 的 BN 状态。

建议优先使用 `SyncBatchNorm`；或者至少恢复 `broadcast_buffers=True`。

3. 随机限量 validation 会严重破坏数据局部性

[DistributedReplacementSampler](/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/data/dataloader.py:613) 在整个 grid sample 空间随机抽样，而 [StandardGridDataset](/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/data/dataloader.py:727) 每次 cache miss 都会解压整个 source。

对当前实际 validation cache 分析：

- 1,000 个 source；
- 1,938,000 个 grid sample；
- 完整顺序验证约为每个 rank 加载 1,000 次 source；
- `max_eval_batch_num=10`、batch size 256 时，每个 rank 只抽 2,560 个样本，却出现约 2,552 次 source 加载；
- 8 卡约需要解压 20,416 次 source，反而比完整验证的约 8,000 次更差。

这基本可以解释为什么限制 validation batch 后仍然很慢。

建议改为“按 source 抽样并批量抽取该 source 的多个 grid”，或者将随机索引按 source 分组后再读取。

4. DDP validation 没有按 source 分片

[DistributedNoPaddingSampler](/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/data/dataloader.py:600) 按 grid sample 索引交错分片。因为 manifest 是 source-major 排列，最终每个 rank 仍然会访问几乎全部 source。

更合理的是：

```text
先把 source 分给不同 rank
→ 每个 rank 只处理自己的 source
→ source 内遍历 grid
```

这样 8 卡 validation 理论上可以让每个 source 只解压一次，而不是每卡一次。

## 数据和缓存问题

5. 每个 rank 都会完整解压 grid manifest

[StandardGridDataset.__init__](/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/data/dataloader.py:657) 会把压缩 NPZ 中的全部数组读入内存，压缩 NPZ 无法真正 mmap。

当前 validation manifest：

- 磁盘压缩后约 1.6 MB；
- 解压后约 386 MB；
- 8 卡约占 3.09 GB 主机内存。

其中 `response_bin` 单独约 310 MB，目前使用 `int64`，但取值只有 `-1～9999`，使用 `int16` 即可降到约 77.5 MB。

另外 `q` 可以由 mask 推导，`m_line` 与 `is_positive` 当前生成逻辑基本重复，`track_relation` 没有被评估使用。

6. `_build_grid_manifest` 峰值内存偏高

[_build_grid_manifest](/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/data/dataloader.py:829) 先把每个 source 的数组保存到 Python list，最后再 `np.concatenate`。合并期间新旧数组同时存在，峰值内存可能接近最终数据的两倍。

数据量增加到数万 source、帧步进变为 1 时，有 OOM 风险。建议使用：

- 预先计算长度并一次性分配；
- 按 source 分片保存；
- 或使用可追加、可 mmap 的格式。

7. cache 无法可靠感知原始数据内容变化

cache key 包含路径、source ID 和配置，但没有包含原始文件的大小、mtime 或内容摘要。如果同名 `data.npz` 被重新生成，旧 cache 仍可能命中。

随机抽 10 条验证只能降低风险，不能可靠发现少量 source 的变化。建议把文件大小和 `mtime_ns` 纳入 fingerprint。

8. 每次训练都会提前生成 test grid

[prepare_data_artifacts](/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/data/dataloader.py:1026) 同时生成 validation 和 test cache，但训练过程只使用 validation。

首次运行时这会把启动耗时和缓存写入量接近翻倍。test grid 可以在最终独立评估时按需生成。

## 配置与可复现性

9. 配置校验缺失较多

[base.py](/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/configs/base.py:106) 当前会接受以下非法值：

- `log_interval_steps=0`
- `eval_interval_steps=0`
- `checkpoint_interval_steps=0`
- `eval_batch_size_per_gpu=0`
- `num_workers=-1`
- 负数的采样权重
- `range_bins < range_block_width`
- 与固定输入契约不一致的 `input_channels`

前三项会直接导致取模除零，`range_bins < range_block_width` 会在生成起始位置时触发 `IndexError`。

还应补充学习率、dropout、optimizer beta、采样权重、span/margin、hidden dimension 等参数的范围校验。

10. 已有 split manifest 会忽略新的划分配置

[_build_or_load_split](/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/data/dataloader.py:782) 发现 manifest 后直接复用，没有确认当前的：

- `source_sample_limit`
- `split_seed`
- train/validation/test fraction
- 数据根目录

是否仍与旧 manifest 一致。

因此通过 `--set split_seed=...` 修改配置后，resolved config 显示的是新值，实际却可能仍在使用旧划分。建议在 split manifest 内保存配置摘要并强制校验。

11. source 选择顺序不能跨机器复现

[discover_sources](/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/data/dataloader.py:68) 按文件系统返回顺序取前 N 个。这样速度快，但即使 `split_seed` 相同，不同机器、不同文件系统也可能选到不同的 N 个 source。

如果需要跨集群复现实验，建议生成一次轻量 source 索引文件，后续直接读取索引，而不是每次排序目录。

12. 限量 validation 不适合直接选择 best checkpoint

有放回随机采样确实能让每次 validation 看到不同样本，但 [trainer.py](/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/train/trainer.py:269) 会直接用这次随机指标更新 best checkpoint。

不同 checkpoint 用不同样本评估，指标并不严格可比；恢复训练后随机 evaluation 序列也不会延续。

建议：

- 固定一份小型验证子集用于选择 best checkpoint；
- 随机有放回验证只用于观察训练趋势；
- 训练结束后进行一次完整 validation/test。

## 训练性能问题

13. 每个 optimizer step 都执行全局 barrier

[trainer.py](/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/train/trainer.py:315) 每一步都调用 `barrier()`。DDP backward 已经完成梯度同步，这个 barrier 通常是多余的，会增加多卡等待时间。

14. loss 中存在设备到主机同步

[losses.py](/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/train/losses.py:81) 的：

```python
if bool(valid_block.any()):
```

会让 GPU/NPU 等待并把结果同步回 CPU。可以使用无分支的 `sum / clamp_min(count, 1)` 实现。

15. `torch.compile` 与梯度累积组合时可能失去 `no_sync`

[trainer.py](/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/train/trainer.py:217) 通过 `isinstance(model, DistributedDataParallel)` 判断是否使用 `no_sync()`。

模型经过 `torch.compile` 后外层通常成为 `OptimizedModule`，判断可能失败，导致每个 microbatch 都进行梯度通信。结果一般仍正确，但梯度累积性能会明显下降。

16. `steps/sec` 计算错误

[trainer.py](/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/train/trainer.py:255) 使用：

```python
steps_per_sec = 1.0 / elapsed
```

但 `elapsed` 是整个日志区间累计时间。默认每 20 step 打印一次时，吞吐量会被低估约 20 倍，应使用“区间 step 数 / elapsed”。

## 无用或冗余代码

- [postprocess.py](/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/eval/postprocess.py:1) 的候选线去重逻辑没有被任何入口调用；其中去重还是 O(N²)。
- `track_relation` 被生成、缓存、读取并传到设备，但 evaluator 没有使用。
- validation batch 中的 `source_index/time_start/range_start` 被移动到 NPU，但指标计算不使用。
- 训练采样返回的 `H/track_relation/time_start/range_start` 最终没有进入 batch。
- `_SourcePool.source_cycle` 只递增，从未被读取。
- `DataArtifacts.test_manifest_path` 在训练启动过程中生成，但训练没有消费。
- README 仍声称每个 run 目录保存 validation/test grid，已经与当前全局 cache 实现不一致。

## 测试状况

整个目录可以通过 Python `compileall`，没有发现语法错误。但 [tests](/mnt/host-model/weixc/code/LineTracker/method/SimpleCNN/tests) 目前只有 `.gitkeep`，没有自动化测试，环境中也没有配置 `pytest/ruff/mypy`。

建议最先补充以下回归测试：

1. 独立 validation/test 能正确找到全局 cache；
2. 标量标签计算、向量化标签计算和 cache 结果一致；
3. 修改 split/cache 相关参数后能正确失效；
4. DDP 下 BN 状态及 validation 指标一致；
5. source-aware sampler 不重不漏；
6. checkpoint 恢复后 optimizer、scheduler、sampler 状态正确。

优先修复顺序建议是：独立评估路径 → BatchNorm/DDP → validation source-locality → manifest 内存格式 → split/config 校验。