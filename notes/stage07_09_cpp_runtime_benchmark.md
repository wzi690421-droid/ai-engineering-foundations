# 阶段 07～09：C++ Runtime 与可信 Benchmark

## 主数据流

```text
图片路径
  → OpenCV读取、RGB转换、resize、归一化、NCHW
  → ONNX Runtime C++ Session
  → logits
  → 稳定Softmax与Top-K
  → 原始延迟CSV和汇总报告
```

## C++ Runtime 关键点

- `OnnxModel` 长期拥有 `Ort::Env`、`SessionOptions`、`Session`、节点名称和输入契约。
- 动态 Batch 只改变输入形状的第0维；通道、高、宽必须满足模型契约。
- Tensor 输入要求连续内存、正确 dtype、shape 和元素数量。
- ORT 输出离开局部作用域后会销毁，因此需要复制到自有 `std::vector<float>`。
- CMake 把公共实现组成 `inference_core`，正确性程序与 benchmark 程序分别链接。

## Benchmark 分层

### 纯推理

只测 `OnnxModel::run()`，包含 Tensor 包装、ORT 推理和输出复制；不测模型加载、图片预处理、后处理和CSV写入。

### 端到端

```text
图片读取/预处理 → 推理 → Softmax/Top-K
```

分段之和必须与同一轮端到端时间对应。CSV写入必须放在计时结束之后。

## 指标含义

- mean：反映总时间和平均吞吐，但容易被极端值影响。
- p50：典型请求延迟。
- p95：大多数请求的尾延迟，但看不到最慢5%的细节。
- throughput：处理图片总数除以总时间，不是独立于 mean 的第二份证据。
- Max RSS：整个进程生命周期的物理内存峰值，不等于模型本身内存。

## 实验原则

1. 正式计时前 warm-up，排除第一次推理的冷启动开销。
2. 新旧版本使用同一模型、输入、Batch、线程、编译模式和 Runtime 配置。
3. 性能测试顺序运行，不能并行争抢 CPU。
4. 重复多轮并保留每一次原始延迟。
5. 不凭感觉删除异常值；先定位、解释，并同时报告 mean、p50、p95 和最大值。
6. 优化前后先验证 logits 或预测结果仍然对齐。

## Batch 结论

- 小 Batch 更适合低响应延迟。
- 大 Batch 能分摊固定开销并提高吞吐，但单批延迟增加，在线系统还会产生凑 Batch 的排队时间。
- “单张有效计算时间 = Batch总计算时间 / 图片数”是吞吐指标，不是单张在线请求的返回时间。

## 当前基线

- Batch=1 纯推理 p50 约 `0.034 ms`。
- Batch=8 纯推理平均吞吐约 `9.4万 images/s`，约为 Batch=1 的3.25倍。
- Batch=1 热缓存端到端典型 p50 约 `0.109 ms`。
- 端到端时间约60%在图片读取与预处理、38%在推理、2%在后处理。
- 进程峰值内存约72 MiB；Batch=1与8差异不明显。

详细数据见：`project/onnxruntime_cpp/results/fp32_baseline_report.md`。

## 易错点

- `git add -N` 只建立 intent-to-add，不保存文件内容，不能作为恢复点。
- 时间占比不能推导内存占比。
- 单轮结果不能证明优化成功。
- 模型文件大小不能代表进程运行内存。
- 热缓存图片读取不能代表首次冷磁盘读取。

## 自测

1. 为什么 warm-up 不能消除运行中途的系统调度抖动？
2. 为什么 Batch=8 的单张有效计算时间不是在线响应时间？
3. mean 与 p50 冲突时应怎样保留和解释证据？
4. 如何测量加载模型前后的增量内存，而不是只看进程 Max RSS？
5. 比较两个优化版本时，哪些变量必须保持一致？
