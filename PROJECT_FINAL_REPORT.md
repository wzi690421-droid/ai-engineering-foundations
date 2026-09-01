# AI Engineering Foundations 最终报告

日期：2026-09-01
状态：基础能力阶段完成，功能冻结

## 1. 项目定位

本项目使用 CIFAR-10 小模型作为低成本负载，完成了一条从算法理解到训练、
模型导出、C++ Runtime、性能测量、CPU 量化、CUDA Kernel 和框架扩展的工程链路。

它的价值不是提供生产级视觉模型，而是证明能够回答以下问题：

1. 模型怎样训练、验证并保存为可信实验；
2. 模型怎样脱离 PyTorch，在 ONNX Runtime C++ 中运行；
3. 怎样定义测量边界并保存可复查的性能证据；
4. 怎样用 Profiler 解释优化为什么有效或无效；
5. 怎样把 CUDA Kernel 接入 PyTorch，而不破坏设备、Stream 和 Tensor 契约。

## 2. 交付成果

| 能力阶段 | 主要交付 | 状态 |
|---|---|---:|
| C++ 工程 | 日志解析、分析器、CMake、CTest、GDB、ASan | 完成 |
| 神经网络基础 | NumPy MLP、反向传播、梯度检查 | 完成 |
| PyTorch 训练 | Fashion-MNIST、Checkpoint、学习率与 Batch 实验 | 完成 |
| CNN 可信实验 | CIFAR-10、残差块、增强对照、冻结测试、错误分析 | 完成 |
| ONNX Runtime | 动态 Batch、跨后端对齐、独立图片推理 | 完成 |
| C++ Runtime | 图片预处理、Session 推理、Softmax/Top-K | 完成 |
| 性能工程 | warm-up、重复实验、p50/p95、吞吐与阶段计时 | 完成 |
| CPU 优化与量化 | 线程/图优化、Profiler、S8S8/U8S8 对照 | 完成 |
| CUDA | 访存、Reduction、GEMM、cuBLAS 和 Nsight Compute | 完成 |
| 框架接入 | `torch.ops.ai_infra.gemm` C++/CUDA Extension | 完成 |

## 3. 2026-09-01 收尾验收

以下结果是在当前工作区重新执行得到，而不是只引用历史记录。

| 验收项 | 结果 |
|---|---|
| C++ 日志项目 | CMake 构建成功，3/3 CTest 通过 |
| NumPy 阶段 | 14/14 Pytest 通过 |
| Python 源码 | week03、week04 全部通过字节码编译检查 |
| ONNX Runtime C++ | `inference_core`、`ort_inspect`、`ort_benchmark`、`ort_pipeline_benchmark` 构建成功 |
| 真实图片推理 | CIFAR-10 猫图片 Top-1 为 `cat`，概率 `76.37%` |
| CUDA Reduction | CPU/GPU 总和一致；Warp Shuffle 首趟相对 Shared Memory 版约 `1.321×` |
| CUDA GEMM | 结果与 cuBLAS 对齐；本次 cuBLAS 相对 tiled Kernel 约 `7.512×` |
| PyTorch Extension | CPU/CUDA、边界、opcheck、非默认 Stream 和输入拒绝测试全部通过 |

Pytest 需要使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，因为系统 ROS 2 的 Pytest
插件会被自动发现，但它不属于本项目环境。

### 收尾时环境快照

| 组件 | 版本 |
|---|---|
| 系统 | Linux 7.0.0-30-generic |
| CPU | Intel Core i7-12700H，20 个逻辑 CPU |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop，4 GiB，Compute Capability 8.6 |
| G++ / CMake / OpenCV | 13.3.0 / 3.28.3 / 4.6.0 |
| ONNX Runtime | 1.28.0 |
| 默认 `.venv` | Python 3.12、NumPy 2.4.4、PyTorch 2.13.0 CPU |
| CUDA Toolkit | 13.2 |
| Extension 验收环境 | PyTorch 2.8.0+cu128、CUDA 12.8 |

PyTorch Extension 使用独立环境，是因为已有 `.so` 与构建时的 Python、PyTorch、
CUDA 和 C++ ABI 绑定；它不能直接加载到默认 `.venv`。

## 4. 代表性工程结论

### ONNX Runtime FP32

- 小模型的端到端主要开销来自图片读取和预处理，不是模型计算；
- Batch 增大能够提高吞吐，但会增加单次请求延迟并引入凑 Batch 的排队时间；
- 微秒级 Benchmark 必须同时保存原始值、p50、p95 和多轮结果。

完整证据见 [`project/onnxruntime_cpp/results/fp32_baseline_report.md`](project/onnxruntime_cpp/results/fp32_baseline_report.md)。

### INT8 量化

- U8S8 模型体积约为 FP32 的 `34.74%`；
- 验证准确率相对 FP32 下降 `0.20` 个百分点；
- 当前小模型在本机 CPU 上，U8S8 的延迟仍高于 FP32；
- 因此“量化后更小”与“量化后更快”必须分别验证。

完整证据见 [`project/onnxruntime_cpp/results/stage11_quantization_report.md`](project/onnxruntime_cpp/results/stage11_quantization_report.md)。

### CUDA 与框架扩展

- 合并访存、Shared Memory Tiling 和 Warp Shuffle 的收益必须通过计时与 Profiler 共同判断；
- 手写教学 Kernel 明显慢于 cuBLAS 或 `torch.matmul` 是合理结果；
- 框架接入的核心是正确维护 Tensor 约束、CUDA Stream、设备和内存所有权，而不是包装一层 Python API。

## 5. 复现命令

### C++ 与 NumPy

```bash
cmake -S exercises/week01/log_parser_project \
      -B exercises/week01/log_parser_project/build
cmake --build exercises/week01/log_parser_project/build --parallel
ctest --test-dir exercises/week01/log_parser_project/build --output-on-failure

cd exercises/week02
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../../.venv/bin/python -m pytest -q \
    day07_assessment/test_mlp_core.py numpy_mlp/test_model.py
```

### ONNX Runtime C++ 与 CUDA

详细依赖、路径和命令分别见：

- [`project/onnxruntime_cpp/README.md`](project/onnxruntime_cpp/README.md)
- [`project/cuda_kernels/README.md`](project/cuda_kernels/README.md)
- [`project/cuda_kernels/pytorch_extension/README.md`](project/cuda_kernels/pytorch_extension/README.md)

## 6. 已知限制

1. `third_party/`、`.venv/`、模型权重、ONNX 文件和构建目录是本地依赖，不进入 Git；
2. CIFAR-10 `SmallCNN` 很小，CPU INT8 结论不能外推到 Transformer 或其他硬件；
3. CUDA Kernel 以教学和性能分析为目标，不替代 cuBLAS、CUTLASS 或生产 Kernel；
4. PyTorch Extension 的已有二进制绑定 Python、PyTorch、CUDA 和 ABI，换环境后需要重新构建；
5. 本项目没有实现在线服务、连续批处理、多 GPU 通信或集群调度。

## 7. 与后续项目的关系

后续主项目为 `mini-infer-runtime`。本仓库提供它所需的底层基础：

```text
C++/CMake与调试
  → ONNX Runtime与Benchmark
  → CUDA线程、访存、同步与Profiler
  → PyTorch自定义算子
  → mini-infer-runtime中的调度、KV Cache、真实推理后端与多GPU系统
```

因此本项目不是被废弃，而是作为基础作品集冻结；推理系统和 GPU 集群能力在新项目继续发展。
