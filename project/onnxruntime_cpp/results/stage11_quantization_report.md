# ONNX Runtime CPU 静态 INT8 量化报告

日期：2026-08-20

## 结论先行

对当前 CIFAR-10 `SmallCNN`、Intel Core i7-12700H 和 ONNX Runtime 1.28.0 `CPUExecutionProvider`：

- S8S8 与 U8S8 都把模型从 `114019` bytes 缩小到约 `39.6 KB`，体积减少约 `65.2%`。
- 两种量化模型的验证准确率都是 `70.98%`，相对 FP32 的 `71.18%` 下降 `0.20` 个百分点。
- U8S8 修复了 S8S8 第一层未融合的问题，但正式 benchmark 中仍慢于 FP32。
- 追求当前 CPU 最低延迟应部署 FP32；只有模型体积比延迟更重要时才选择 U8S8。

量化不是自动加速按钮。模型大小、精度和目标硬件上的实际延迟必须分别验证。

## 对象与环境

- 模型：CIFAR-10 `SmallCNN`
- 输入：动态 Batch、FP32、NCHW `[N, 3, 32, 32]`
- FP32 模型：`small_cnn_dynamic_batch.onnx`
- 量化格式：静态 QDQ
- CPU：Intel Core i7-12700H
- Runtime：ONNX Runtime 1.28.0 CPU Execution Provider
- 正式 benchmark：固定 P-core 逻辑 CPU `0-11`、`intra_op=4`、图优化 `all`、profiler 关闭

纯推理测量包含 Tensor 包装、ONNX Runtime 推理和输出复制；不包含模型加载、图片读取、预处理、Top-K 和 CSV 写入。

## 校准与量化配置

校准数据来自固定训练集子集，不使用验证集或测试集：

- 校准图片：512 张
- Batch：32，共 16 批
- 预处理：固定评估预处理，无随机裁剪和翻转
- 校准方法：MinMax
- 权重量化：S8，per-tensor
- 激活对照：S8 与 U8
- `reduce_range=false`

校准不会训练或修改权重。它让 FP32 模型运行代表性输入，统计各中间激活张量的范围，并生成固定的 scale 和 zero-point。

S8S8 与 U8S8 的输入量化刻度相同，只是整数编码平移：

| 模型 | input scale | input zero-point | ReLU scale | ReLU zero-point |
|---|---:|---:|---:|---:|
| S8S8 | 0.0161400679 | -5 | 0.0186664648 | -128 |
| U8S8 | 0.0161400679 | 123 | 0.0186664648 | 0 |

因此 `q_u8 = q_s8 + 128` 时表示的近似实数基本不变，但执行器能够选择不同的量化内核和数据布局。

## 模型大小与正确性

| 模型 | 文件大小（bytes） | FP32占比 | 验证准确率 | 相对FP32变化 |
|---|---:|---:|---:|---:|
| FP32 | 114019 | 100% | 71.18% | — |
| S8S8 QDQ | 39637 | 34.76% | 70.98% | -0.20个百分点 |
| U8S8 QDQ | 39611 | 34.74% | 70.98% | -0.20个百分点 |

5000 张验证图片上，两种 INT8 相对 FP32 都有 79 张 Top-1 预测发生变化：

- FP32 正确、INT8 错误：32 张
- FP32 错误、INT8 正确：22 张
- 两者都错但错误类别不同：25 张

净减少正确数量为 `32 - 22 = 10`，即 `10 / 5000 = 0.20` 个百分点。S8S8 与 U8S8 之间没有 Top-1 预测差异；两者平均 logits 绝对差约 `1.9e-6`。

验证记录：

- `exercises/week04/cifar10_cnn/experiments/small_cnn_augmentation_on/run_20260812_165446/quantization/validation_metrics.json`
- `exercises/week04/cifar10_cnn/experiments/small_cnn_augmentation_on/run_20260812_165446/quantization/validation_metrics_u8s8.json`

## 正式性能结果

Batch 1 和 8 每组先预热 100 次、正式测量 2000 次并重复 5 轮；Batch 500 每组预热 50 次、正式测量 200 次并重复 5 轮。下表报告 5 轮 run-level 指标的中位数。

| Batch | 模型 | mean（ms） | p50（ms） | p95（ms） | 吞吐（images/s） |
|---:|---|---:|---:|---:|---:|
| 1 | FP32 | 0.024599 | 0.024307 | 0.026924 | 40652 |
| 1 | S8S8 | 0.071049 | 0.070629 | 0.078022 | 14075 |
| 1 | U8S8 | 0.031300 | 0.031134 | 0.034790 | 31949 |
| 8 | FP32 | 0.092184 | 0.091532 | 0.097145 | 86783 |
| 8 | S8S8 | 0.496156 | 0.496004 | 0.522393 | 16124 |
| 8 | U8S8 | 0.186029 | 0.184937 | 0.206463 | 43004 |
| 500 | FP32 | 5.440615 | 5.383706 | 5.564149 | 91901 |
| 500 | U8S8 | 10.751725 | 10.676458 | 11.066676 | 46504 |

U8S8 相对 S8S8：

- Batch 1 的 p50 约快 `2.27` 倍。
- Batch 8 的 p50 约快 `2.68` 倍。

U8S8 相对 FP32：

- Batch 1 的 p50 高约 `28%`。
- Batch 8 与 500 的 p50 都接近 FP32 的 `2` 倍。

Batch 从 8 增加到 500 后，FP32 与 U8S8 的单张等效 p50 都只小幅下降，二者比例没有趋向交叉。这说明当前主要损失不是能被大 Batch 摊薄的固定启动开销。

完整 40 行正式记录见 [`stage11_quantization_benchmark.csv`](stage11_quantization_benchmark.csv)。

## 优化图与 Profiler 证据

### FP32

```text
NCHWc Conv+ReLU
→ NCHWc MaxPool
→ NCHWc Conv+ReLU
→ NCHWc MaxPool
→ 一次 ReorderOutput
→ Gemm
```

FP32 从第一层到第二层保持连续的 CPU 友好布局。

### S8S8

```text
输入 Q/DQ
→ 第一层普通 FP32 Conv
→ 普通 MaxPool
→ Transpose
→ 第二层 QLinearConv
→ NhwcMaxPool
→ Transpose
→ QGemm
```

第一层 QDQ 没有融合成整数卷积，混合执行与布局转换导致明显退化。

### U8S8

```text
输入 Quantize
→ Transpose
→ 第一层 QLinearConv
→ NhwcMaxPool
→ 第二层 QLinearConv
→ NhwcMaxPool
→ Transpose
→ QGemm
→ 输出 Dequantize
```

U8S8 成功让两层卷积都进入量化内核，但 Batch 8 的节点对照仍显示：

| 节点 | FP32（μs） | U8S8（μs） | 解释 |
|---|---:|---:|---|
| 第一层卷积 | 30.55 | 78.84 | 当前首层量化内核更慢 |
| 第二层卷积 | 41.26 | 28.90 | 量化取得约30%收益 |
| 第一层池化 | 9.94 | 20.39 | 两条路径的布局与内核不同 |
| 第二层池化 | 7.23 | 9.13 | U8S8略慢 |
| 全连接 | 9.65 | 8.19 | U8S8略快 |
| 布局转换 | Reorder 6.42 | 两次 Transpose 68.58 | U8S8的主要损失之一 |

Profiler 只用于解释节点、融合和相对热点。开启 profiler 会插入事件记录，并且 U8S8 节点数更多，因此 profiler 下的总延迟不能代替关闭 profiler 的正式 benchmark。

## 工程判断

| 生产目标 | 当前选择 | 原因 |
|---|---|---|
| 最低 CPU 延迟 | FP32 | NCHWc 连续路径在 Batch 1、8、500 都更快 |
| 最小模型体积 | U8S8 | 体积减少约65.3%，准确率只下降0.20个百分点，且快于S8S8 |
| 证明“INT8在CPU上更慢” | 不成立 | 结论只适用于当前模型、硬件、Runtime和执行路径 |

更大的计算密集模型、不同硬件或不同 Execution Provider 可能从 INT8 获得不同收益，必须重新完成正确性、精度、延迟和 profiler 闭环。

## 限制

1. 模型只有约 114 KB，FP32 权重可轻易进入 CPU cache，不能代表大型 CNN 或 Transformer。
2. Batch 500 使用同一张图片复制为 500 个 Tensor 样本，只用于性能路径，不用于准确率判断。
3. 本报告比较的是纯推理边界，不包含图片预处理和在线系统凑 Batch 的排队时间。
4. Profile 节点时间带有测量侵入性，正式部署选择以 profiler 关闭时的数据为准。
5. 当前只测试 ONNX Runtime CPU Execution Provider，没有测试 OpenVINO、TensorRT 或其他后端。

## 验收

阶段 11 口头验收：`4.25/5`。

已掌握静态校准、scale/zero-point、S8/U8编码、QDQ与实际整数内核的区别，以及准确率—模型大小—延迟的联合判断。当前主要缺口是进一步提高技术表达精度，明确区分实验事实、Profiler推断和不能外推的普遍结论。
