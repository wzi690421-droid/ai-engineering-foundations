# ONNX Runtime C++ Inference

这个子项目把 PyTorch 导出的 CIFAR-10 ONNX 模型接入 C++，并把正确性检查、
纯推理 Benchmark 和端到端阶段计时拆成三个可执行程序。

## 数据流

```text
图片路径
  → OpenCV读取、RGB转换、NCHW与归一化
  → ONNX Runtime Session
  → logits
  → 稳定Softmax与Top-K
```

## 目录

- `include/`、`src/`：Runtime、预处理、后处理和 Benchmark 源码；
- `scripts/`：批量汇总实验结果的 Python 脚本；
- `results/`：原始 CSV、FP32 报告和 INT8 报告；
- `build/`：本机构建产物，不进入 Git。

## 依赖

- CMake 3.20+
- 支持 C++17 的编译器
- OpenCV 4
- ONNX Runtime C++ SDK 1.28.0

默认 SDK 路径为：

```text
third_party/onnxruntime-linux-x64-1.28.0
```

可以在配置时用 `-DONNXRUNTIME_ROOT=/path/to/onnxruntime` 覆盖。

## 构建

在仓库根目录执行：

```bash
cmake -S project/onnxruntime_cpp \
      -B project/onnxruntime_cpp/build
cmake --build project/onnxruntime_cpp/build --parallel
```

构建目标：

- `ort_inspect`：真实图片正确性与 Top-K；
- `ort_benchmark`：图片只预处理一次，重复测量 `OnnxModel::run()`；
- `ort_pipeline_benchmark`：测量预处理、推理和后处理的端到端时间。

## 正确性示例

```bash
MODEL=exercises/week04/cifar10_cnn/experiments/small_cnn_augmentation_on/run_20260812_165446/exports/small_cnn_dynamic_batch.onnx
CONFIG=exercises/week04/cifar10_cnn/experiments/small_cnn_augmentation_on/run_20260812_165446/config.json
IMAGE=exercises/week04/cifar10_cnn/runtime_samples/cifar10_test_00000_cat.png

project/onnxruntime_cpp/build/ort_inspect \
    "$MODEL" "$CONFIG" 3 "$IMAGE"
```

预期输入/输出形状为 `[1, 3, 32, 32]` 和 `[1, 10]`，该固定样本的
Top-1 应为 `cat`。

## Benchmark 接口

```text
ort_benchmark MODEL CONFIG WARMUP MEASURED REPEATS OUTPUT_DIR \
    INTRA_OP_THREADS GRAPH_OPTIMIZATION PROFILING IMAGE...
```

- `GRAPH_OPTIMIZATION`：`disabled`、`basic`、`extended` 或 `all`；
- `PROFILING`：`off` 或 `profile`；
- `INTRA_OP_THREADS=0` 表示使用 ONNX Runtime 默认值。

端到端程序接口：

```text
ort_pipeline_benchmark MODEL CONFIG WARMUP MEASURED OUTPUT_CSV TOP_K IMAGE...
```

正式比较应关闭 Profiler、固定线程与 CPU 亲和性、先预热、重复多轮，
同时保留原始 CSV、p50 和 p95。

## 报告

- [`results/fp32_baseline_report.md`](results/fp32_baseline_report.md)
- [`results/stage11_quantization_report.md`](results/stage11_quantization_report.md)
