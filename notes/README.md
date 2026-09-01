# AI 工程学习笔记

这套笔记不按日期记录，而是按能力阶段整理。目标不是保存所有代码细节，而是在复盘时快速恢复“为什么这样做、数据如何流动、哪里容易错、如何证明结果可信”。

## 总路线

```text
阶段 01：先能写出可靠的小型 C++ 工程
  → 阶段 02：理解神经网络前向、损失、梯度和更新
  → 阶段 03：把数学映射到 PyTorch 训练工程
  → 阶段 04：用 CNN 处理真实图像并理解残差网络
  → 阶段 05：让训练结果可复现、可比较、可信
  → 阶段 06：导出 ONNX，脱离 PyTorch 完成推理
  → 阶段 07～09：用 C++ 承接 Runtime，并建立可信性能基线
  → 阶段 10～11：用 Profiler 优化 CPU，并完成 INT8 精度—性能取舍
  → 阶段 12～13：建立 CUDA Kernel、Profiler 与 PyTorch 扩展能力
```

## 阶段索引

| 阶段 | 核心问题 | 主要成果 | 验收 |
|---|---|---|---:|
| [01：C++ 工程基础](stage01_cpp_foundation.md) | 如何写、测、调一个小型 C++ 模块 | 日志分析器、CMake、CTest、GDB、ASan | 3/5 |
| [02：NumPy 神经网络](stage02_numpy_mlp.md) | 神经网络内部究竟如何计算 | 两层 MLP、反向传播、梯度检查 | 3/5 |
| [03：PyTorch 训练](stage03_pytorch_training.md) | 框架如何替代手写梯度并组织训练 | Fashion-MNIST 训练、checkpoint、学习率实验 | 3/5 |
| [04：CNN 与残差](stage04_cnn_resnet.md) | 为什么 CNN 更适合图像，残差如何工作 | CIFAR-10 CNN、残差块、公平对照 | 3.5/5 |
| [05：可信训练](stage05_reliable_training.md) | 一个准确率为什么值得相信 | 数据隔离、实验记录、严格续训、冻结测试 | 4.8/5 |
| [06：ONNX Runtime](stage06_onnx_runtime.md) | 模型如何脱离 PyTorch 运行 | 动态 ONNX、跨后端对齐、独立图片推理 | 3.55/5 |
| [07～09：C++ Runtime 与 Benchmark](stage07_09_cpp_runtime_benchmark.md) | 如何完成 C++ 推理并可信测量性能 | 动态 Batch、端到端推理、原始 CSV、FP32 基线报告 | 07：4.3/5；08：3.7/5；09：3.7/5 |
| [10～11：CPU优化与INT8量化](stage10_11_cpu_optimization_quantization.md) | 如何用证据优化并判断量化是否值得部署 | 线程/图优化、Profiler、S8S8/U8S8、精度—体积—延迟报告 | 10：4.4/5；11：4.25/5 |
| [12～13：CUDA Kernel与PyTorch扩展](stage13_cuda_kernel_pytorch_extension.md) | 如何从线程、访存和同步解释性能，并安全接入框架 | Reduction、GEMM、cuBLAS对照、Nsight Compute、`torch.ops`扩展 | 完成 |

## 推荐复盘方式

每次不要从头抄代码，按下面顺序复习：

1. 先闭卷画出该阶段的主数据流。
2. 阅读“关键知识”和“易错点”，判断自己能否解释原因。
3. 回答文末自测，不会的内容再回到对应代码。
4. 至少亲手运行一次该阶段的核心命令。
5. 能修改一个小需求并保持测试通过，才算真正掌握。

评分含义：`3/5` 表示在工具和适量提示下可以完成；不代表已经能够闭卷从空目录重建。后续阶段会持续复测旧知识，而不是反复重学整套内容。
