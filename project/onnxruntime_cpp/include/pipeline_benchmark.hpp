#pragma once

#include "image_preprocessor.hpp"
#include "inference_benchmark.hpp"
#include "onnx_model.hpp"

#include <cstddef>
#include <string>
#include <vector>

// 保存一次端到端推理中各阶段的延迟。
struct PipelineLatencySample {
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
    double end_to_end_ms;
};

// 保存全部正式测量的分段延迟。
struct PipelineBenchmarkResult {
    std::vector<PipelineLatencySample> samples;
};

// 重复执行完整图片分类流程。
PipelineBenchmarkResult benchmarkPipeline(
    OnnxModel& model,
    const ImagePreprocessor& preprocessor,
    const std::vector<std::string>& image_paths,
    const std::vector<std::string>& class_names,
    std::size_t top_k,
    const InferenceBenchmarkConfig& config
);

// 保存每次端到端分段延迟。
void savePipelineSamplesCsv(
    const std::string& output_path,
    std::size_t batch_size,
    const InferenceBenchmarkConfig& config,
    const PipelineBenchmarkResult& result
);
