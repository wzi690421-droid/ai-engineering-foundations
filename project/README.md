# Project Index

这里保存基础阶段的两个主要工程成果。

## ONNX Runtime C++

路径：[`onnxruntime_cpp/`](onnxruntime_cpp/README.md)

完成真实图片预处理、ONNX Runtime C++ 推理、Softmax/Top-K、纯推理
Benchmark、端到端分段计时、CPU 线程/图优化和 INT8 对照实验。

## CUDA Kernels

路径：[`cuda_kernels/`](cuda_kernels/README.md)

完成线程映射、合并访存、Reduction、GEMM、cuBLAS 对照、Nsight Compute
分析和 PyTorch C++/CUDA Extension。

两个子项目都已进入功能冻结状态。后续主项目是独立仓库
`mini-infer-runtime`；这里只做复现修复与必要维护。
