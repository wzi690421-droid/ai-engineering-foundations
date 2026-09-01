# 阶段 10～11：CPU 优化、Profiler 与 INT8 量化

## 主闭环

```text
固定正确性与测量边界
  → 单变量比较线程数和图优化
  → benchmark证明是否真的变快
  → profiler解释算子、融合和热点
  → 静态量化与精度验证
  → FP32/S8S8/U8S8公平对照
  → 根据目标选择部署版本
```

## 阶段 10：CPU 推理优化

### Session 配置

`SessionOptions` 必须在创建 `Ort::Session` 前完成配置，因为 Session 初始化时会读取线程数、执行模式、图优化和 profiling 选项。Session 创建后再修改原来的 options 不会回头改变已经创建的 Session。

本阶段把以下参数纳入同一 C++ benchmark：

- intra-op 线程数
- 图优化级别：disabled/basic/extended/all
- profiler 开关与输出前缀
- 固定模型、输入、Batch、warm-up、正式迭代和重复轮数

### 线程实验

- 当前 i7-12700H 是混合架构 CPU，0～11 为 P-core 逻辑 CPU，12～19 为 E-core。
- 对 Batch 1 的极小 CNN，更多线程不一定更快；线程调度和同步可能超过计算收益。
- 多轮与反向顺序实验后，`intra_op=4` 是较稳健选择。
- 只绑定 P-core 后，4 线程 p50 约 `0.02731 ms`，8 线程约 `0.02894 ms`；8 线程仍慢约 `5.9%`。

### 图优化实验

`all` 明显优于 disabled/basic/extended。Profiler 证明差异来自真实计算图变化，而不是参数名字：

- Conv 与 ReLU 融合；
- 卷积和池化进入 NCHWc 路径；
- 中间节点减少；
- 只在卷积区域出口做一次 ReorderOutput。

不能把所有收益只归因于“NCHWc”一个词。融合、布局、节点数量和内核选择同时发生变化；若要隔离某一个原因，需要更严格的专门对照。

### Benchmark 与 Profiler

- benchmark：关闭 profiler，回答端到端测量边界内到底快不快。
- profiler：打开节点追踪，回答时间花在哪里、执行了哪些算子、是否融合。
- profiler 有侵入性；节点更多的模型会记录更多事件，因此 profiler 下的总延迟不能作为正式性能对比。

阶段 10 验收：`4.4/5`。

## 阶段 11：静态 INT8

### 量化公式

```text
x_fp32 ≈ scale × (q - zero_point)
```

- `scale` 决定量化刻度。
- `zero_point` 表示真实数值 0 对应的整数编码。
- S8 与 U8 都有 256 个整数位置；若 scale 相同且编码与 zero-point 同时平移 128，表示的近似实数基本不变。

当前输入参数：

```text
S8：scale=0.0161400679，zero-point=-5
U8：scale=0.0161400679，zero-point=123
```

例如 S8 编码 10 与 U8 编码 138 都表示：

```text
0.0161400679 × 15
```

### 校准

- 固定取训练集 512 张，每批 32 张，共 16 批。
- 使用评估预处理，不使用随机裁剪或翻转。
- MinMax 统计各中间激活张量的观察范围并生成 scale/zero-point。
- 校准不训练、不反向传播、不修改权重。
- 验证集和测试集不能用于确定量化参数，否则评估数据参与模型选择，形成信息泄漏。

### QDQ 不等于所有节点都在做整数计算

QDQ 原图可能写成：

```text
Quantize → Dequantize → Conv → Quantize
```

它本身是可执行图。Runtime 若识别到支持的模式，可以把整段融合成 `QLinearConv`；融合失败时会真的反量化并执行普通浮点 Conv。因此必须查看优化图或 profiler，不能只看模型文件中有没有 INT8 initializer。

### S8S8 与 U8S8

S8S8 优化图中：

- 第一层仍是普通 `Conv`；
- 第二层是 `QLinearConv`；
- 第一层池化走普通路径；
- 中间存在量化、反量化和 Transpose。

U8S8 优化图中：

- 两层都是 `QLinearConv`；
- 两层池化都是 `NhwcMaxPool`；
- 卷积区域内部保持连续 NHWC；
- 入口和出口仍各有一次 Transpose。

U8S8 的图更完整，速度比 S8S8 提高约 2.3～2.7 倍，但仍没有超过 FP32 NCHWc。

## 最终结果

| Batch | 模型 | p50中位数（ms） | 吞吐中位数（images/s） |
|---:|---|---:|---:|
| 1 | FP32 | 0.024307 | 40652 |
| 1 | S8S8 | 0.070629 | 14075 |
| 1 | U8S8 | 0.031134 | 31949 |
| 8 | FP32 | 0.091532 | 86783 |
| 8 | S8S8 | 0.496004 | 16124 |
| 8 | U8S8 | 0.184937 | 43004 |
| 500 | FP32 | 5.383706 | 91901 |
| 500 | U8S8 | 10.676458 | 46504 |

| 模型 | 大小 | 验证准确率 |
|---|---:|---:|
| FP32 | 114019 bytes | 71.18% |
| S8S8 | 39637 bytes | 70.98% |
| U8S8 | 39611 bytes | 70.98% |

Batch 500 的对照说明 U8S8 的损失不是单纯固定启动成本：Transpose、量化和部分算子成本会随 Tensor 规模增长，第一层量化卷积本身在当前路径上也慢于 FP32 NCHWc。

## 部署判断

```text
最低CPU延迟 → FP32
最小模型体积 → U8S8
```

不能推出“INT8在CPU上天生比FP32慢”。当前结论只属于这个极小 CNN、这台 CPU、这个 Runtime 版本、当前 Execution Provider 和测量边界。换模型、硬件或后端必须重做正确性—精度—性能闭环。

详细报告：`project/onnxruntime_cpp/results/stage11_quantization_report.md`。

## 易错点

- 模型变小不等于延迟降低。
- 权重是 INT8 不等于算子一定执行整数计算。
- QDQ 是可执行表示，不是永远不会执行的注释模板。
- 不能把普通 MaxPool 与 NCHWc/NHWC 专用内核的耗时直接解释为“某种精度天生更慢”。
- 不能使用 profiler 下的总延迟替代正式 benchmark。
- Batch 增大能摊薄固定开销，但不能摊薄所有逐元素和布局转换成本。
- 一次负结果不是失败；公平实验能够证明优化不适用于当前目标同样有价值。

## AI 参与边界

- Python 量化调用、验证统计和结果保存由 AI 提供或直接修改。
- 学习者完成了校准、S8/U8、QDQ、benchmark/profiler边界和部署取舍的口头答辩。
- U8S8对照、Batch 500反证和“池化路径不可直接归因”的关键实验判断由学习者提出。

阶段 11 验收：`4.25/5`。主要缺口是技术表达需要更严格地区分直接事实、Profiler推断和不能外推的普遍结论。

## 复盘题

1. 为什么相同 scale、相差128的zero-point可以让S8和U8表示相同实数？
2. 为什么原始QDQ图中存在普通Conv，仍不能直接判断最终执行的是FP32还是INT8？
3. 怎样用优化图和profiler区分“量化未融合”与“量化内核本身较慢”？
4. 为什么Batch 500仍未反超可以否定“当前只是固定开销”的解释，却不能证明所有CPU上的INT8都更慢？
5. 面向最低延迟和最小模型体积，为什么会选择不同部署版本？
