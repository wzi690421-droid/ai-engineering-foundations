#pragma once

#include "onnx_model.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// 控制一次基准测试怎么运行。
struct InferenceBenchmarkConfig {
    std::size_t warmup_iterations;

    std::size_t measured_iterations;
};

// 保存一次完整基准实验的统计结果。
struct InferenceBenchmarkResult {
    double mean_latency_ms;
    double p50_latency_ms;
    double p95_latency_ms;

    double min_latency_ms;
    double max_latency_ms;

    // 最慢数据在原始测量序列中的下标，从0开始。
    std::size_t max_latency_index;

    double throughput_images_per_second;

    // 按实际执行顺序保存每次延迟。
    std::vector<double> latencies_ms;
};

// 对已经完成预处理的数据反复执行模型推理。
InferenceBenchmarkResult benchmarkInference(
    OnnxModel& model,
    std::vector<float>& input_values,
    const std::vector<std::int64_t>& input_shape,
    const InferenceBenchmarkConfig& config
);

// 把原始延迟及本次实验条件保存为CSV。
void saveLatencySamplesCsv(
    const std::string& output_path,
    const InferenceBenchmarkConfig& config,
    std::int64_t batch_size,
    const InferenceBenchmarkResult& result
);

// 把多轮benchmark的汇总指标保存为CSV。
void saveBenchmarkSummaryCsv(
    const std::string& output_path,
    const InferenceBenchmarkConfig& config,
    std::int64_t batch_size,
    const std::vector<InferenceBenchmarkResult>& results
);

void saveTensorValuesCsv(
    const std::string& output_path,
    const TensorResult& result
);
