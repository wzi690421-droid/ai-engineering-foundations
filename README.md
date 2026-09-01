# AI Engineering Foundations

> 状态：基础阶段已完成并冻结（2026-09-01）。后续主项目迁移到
> `mini-infer-runtime`，本仓库保留为可复现的训练、推理和 CUDA 基础作品集。

这个仓库记录了从 C++ 工程基础到 PyTorch、ONNX Runtime、CPU 推理分析、
CUDA Kernel 和 PyTorch 自定义算子的完整学习链路。它不再继续叠加新功能，
后续只接受复现修复、文档修正和必要的兼容性维护。

## 已完成能力

- C++17、CMake、CTest、GDB 和 AddressSanitizer；
- NumPy 两层神经网络、反向传播和梯度检查；
- PyTorch 训练、验证、Checkpoint 和可复现实验；
- CNN、残差网络、数据增强和错误分析；
- PyTorch → ONNX → ONNX Runtime C++ 推理；
- FP32/INT8 正确性、精度、延迟与体积对照；
- CUDA 线程映射、合并访存、Reduction、GEMM 和 Nsight Compute；
- PyTorch C++/CUDA Extension 与框架算子注册。

完整成果、验收结果和限制见 [`PROJECT_FINAL_REPORT.md`](PROJECT_FINAL_REPORT.md)。

## 成果入口

- [`notes/README.md`](notes/README.md)：按能力阶段组织的复盘笔记；
- [`project/onnxruntime_cpp/README.md`](project/onnxruntime_cpp/README.md)：C++ 推理与 Benchmark；
- [`project/cuda_kernels/README.md`](project/cuda_kernels/README.md)：CUDA Kernel 与性能实验；
- [`project/cuda_kernels/pytorch_extension/README.md`](project/cuda_kernels/pytorch_extension/README.md)：PyTorch 自定义算子；
- [`project/onnxruntime_cpp/results/fp32_baseline_report.md`](project/onnxruntime_cpp/results/fp32_baseline_report.md)：FP32 基线报告；
- [`project/onnxruntime_cpp/results/stage11_quantization_report.md`](project/onnxruntime_cpp/results/stage11_quantization_report.md)：INT8 量化报告。

## 目录

- `exercises/`：C++、NumPy、PyTorch 和 CNN 阶段练习；
- `project/onnxruntime_cpp/`：CPU 推理、端到端流水线和 Benchmark；
- `project/cuda_kernels/`：CUDA Kernel、Profiler 实验和 PyTorch Extension；
- `notes/`：阶段复盘；
- `plan/`：本仓库计划归档说明；
- `benchmarks/`、各子项目 `results/`：实验原始数据和报告。

## 复现边界

第三方 SDK、虚拟环境、构建目录、模型权重和 ONNX 文件不进入 Git。
复现 ONNX Runtime 示例需要先准备对应 SDK 与模型；CUDA 示例需要 NVIDIA GPU、
CUDA Toolkit 和匹配的驱动。详细命令和本次验收记录见最终报告及两个子项目 README。
