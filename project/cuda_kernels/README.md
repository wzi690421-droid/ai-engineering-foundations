# CUDA Kernels and Profiling Labs

这个子项目从 CUDA 线程索引开始，逐步完成访存、Reduction、GEMM、Profiler
分析和 PyTorch 自定义算子接入。目标是建立可解释的 GPU 性能分析能力，
不是用教学 Kernel 替代高性能库。

## 源码路线

| 文件 | 主题 |
|---|---|
| `add_one.cu`、`vector_add.cu` | Kernel 启动、Grid-Stride Loop、正确计时与传输 |
| `matrix_add.cu`、`warp_layout.cu` | 二维索引、Warp 与边界线程 |
| `memory_access.cu` | 合并访存与跨步访存对照 |
| `reduction_cuda.cu` | Shared Memory 与 Warp Shuffle Reduction |
| `reduction_max.cu` | Reduction 算子的泛化 |
| `gemm_naive.cu` | 朴素 GEMM 基线 |
| `gemm_memory_access.cu` | GEMM 合并访存 |
| `gemm_tiled.cu` | Shared Memory Tiling |
| `gemm_cublas_compare.cu` | 手写 Kernel 与 cuBLAS 公平对照 |
| `device_info.cu` | 设备、SM、Warp、Block 与 Grid 上限 |
| `pytorch_extension/` | `torch.ops.ai_infra.gemm` C++/CUDA Extension |

## 环境

- NVIDIA GPU 与匹配驱动
- CUDA Toolkit
- 本机验证架构：RTX 3050 Ti Laptop，Compute Capability 8.6

## 代表性构建与验证

在 `project/cuda_kernels/` 目录执行：

```bash
nvcc -std=c++17 -arch=sm_86 -Xcompiler=-Wall,-Wextra \
    src/device_info.cu -o /tmp/device_info
/tmp/device_info

nvcc -std=c++17 -arch=sm_86 -Xcompiler=-Wall,-Wextra \
    src/reduction_cuda.cu -o /tmp/reduction_cuda
/tmp/reduction_cuda

nvcc -std=c++17 -arch=sm_86 -Xcompiler=-Wall,-Wextra \
    src/gemm_cublas_compare.cu -lcublas -o /tmp/gemm_cublas_compare
/tmp/gemm_cublas_compare
```

如果 GPU 架构不是 8.6，需要把 `sm_86` 改为设备对应的目标架构。

正确性验收：

- Reduction 的 CPU/GPU 最终结果一致；
- GEMM 的最大绝对误差满足源码中的容差；
- 性能结果必须在相同输入、预热、迭代次数和计时边界下比较。

## Nsight Compute

Profiler 用来解释执行瓶颈，不替代关闭 Profiler 时的正式计时。分析顺序：

```text
先确认正确性
  → 稳定重复计时
  → 判断计算、访存或同步瓶颈
  → 用Scheduler、Warp Stall、Memory Workload和Source指标验证
  → 修改一个变量后重新对照
```

## PyTorch Extension

构建环境、注册方式和完整验证见
[`pytorch_extension/README.md`](pytorch_extension/README.md)。
