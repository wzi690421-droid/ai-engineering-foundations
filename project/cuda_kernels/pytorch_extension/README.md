# Stage 13.6：PyTorch CUDA Extension

这个扩展将阶段 13.4 的 16×16 Shared Memory tiled GEMM 注册为：

```python
torch.ops.ai_infra.gemm(a, b)
```

数据路径：

```text
Python Tensor
  → PyTorch Dispatcher
  → CUDA 实现
  → 当前 CUDA Stream 上的 gemmTiledKernel
  → PyTorch 管理的输出 Tensor
```

本机使用与 `torch 2.8.0+cu128` 匹配的独立 CUDA 12.8 工具链构建：

```bash
export CUDA_HOME=/home/wz/miniconda3/envs/cuda128-toolchain
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=4

/home/wz/miniconda3/envs/yolo/bin/python setup.py build_ext --inplace
/home/wz/miniconda3/envs/yolo/bin/python test_gemm_extension.py
```

测试覆盖：CPU/CUDA参考结果、非16倍数边界、`opcheck`、非默认CUDA Stream，以及输入约束。
