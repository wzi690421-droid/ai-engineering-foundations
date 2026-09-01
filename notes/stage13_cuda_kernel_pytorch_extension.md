# 阶段13：CUDA Kernel底座与PyTorch集成

## 完成内容

- Warp Shuffle Reduction：减少Shared Memory、Barrier和执行指令。
- 三版GEMM：朴素映射、合并访存、16×16 Shared Memory Tiling。
- 库对照：与cuBLAS及`torch.matmul`保存正确性、延迟和Profiler证据。
- 框架接入：将亲自分析过的tiled GEMM注册为`torch.ops.ai_infra.gemm`。

## 关键性能证据

| 对照 | 结果 | 主要证据 |
|---|---:|---|
| Shared Reduction → Warp Shuffle | `1.251×` | 执行指令下降42.3%，Barrier `6.40 → 4.77`，MIO `3.53 → 0.84` |
| Strided → Coalesced GEMM | `7.746×` | L1TEX Sector下降84.9%，Long Scoreboard下降78.8% |
| Naive → Tiled GEMM | `1.298×` | Global Load Sector下降85.5%，Long Scoreboard下降42.1%，但引入Barrier与MIO压力 |
| Tiled GEMM → cuBLAS | cuBLAS快`7.573×` | cuBLAS用更少线程、更高寄存器复用和ILP获得更高IPC与Issue Slot利用率 |
| PyTorch Extension → `torch.matmul` | `0.475 ms` vs `0.094 ms` | Dispatcher接入不会自动消除Kernel本身与高性能库之间的差距 |

## PyTorch Extension契约

```text
Python Tensor
  → PyTorch Dispatcher
  → CPU或CUDA实现
  → 当前CUDA Stream上的tiled GEMM Kernel
  → PyTorch分配并管理的输出Tensor
```

扩展当前只接受二维、同设备、连续的FP32 Tensor，并检查矩阵内维是否匹配。CUDA实现使用输入Tensor所在设备、PyTorch当前CUDA Stream和PyTorch分配器创建的输出Tensor，不在扩展内部执行强制同步。

## 验收结果

- `torch 2.8.0+cu128`与独立`nvcc 12.8.93`工具链成功构建，目标架构为`sm_86`。
- CPU参考最大绝对误差：`0`。
- CUDA最大绝对误差：`9.918e-5`。
- `torch.library.opcheck`的schema、autograd registration、FakeTensor和AOT dynamic四项均通过。
- 非默认CUDA Stream、非连续输入拒绝及零维度边界均通过。

阶段13的结论不是“手写GEMM胜过库”，而是能够从线程映射、访存、Shared Memory、同步、寄存器复用和Dispatcher一直解释到性能结果，并将同一个Kernel安全接入真实框架。
