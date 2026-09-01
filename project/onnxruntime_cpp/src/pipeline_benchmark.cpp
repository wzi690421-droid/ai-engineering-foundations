#include "pipeline_benchmark.hpp"

#include "classification_postprocessor.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace {

using PipelineClock = std::chrono::steady_clock;


// 把两个时间点的差转换成毫秒。
double durationMilliseconds(
    PipelineClock::time_point start,
    PipelineClock::time_point end
) {
    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

}  // namespace

PipelineBenchmarkResult benchmarkPipeline(
    OnnxModel& model,
    const ImagePreprocessor& preprocessor,
    const std::vector<std::string>& image_paths,
    const std::vector<std::string>& class_names,
    std::size_t top_k,
    const InferenceBenchmarkConfig& config
) {
    if (config.measured_iterations == 0) {
        throw std::invalid_argument(
            "measured_iterations must be positive"
        );
    }

    // 预热完整流程，但不保存时间。
    for (
        std::size_t iteration = 0;
        iteration < config.warmup_iterations;
        ++iteration
    ) {
        ImageBatch batch =
            preprocessor.preprocess(image_paths);

        TensorResult output =
            model.run(batch.values, batch.shape);

        const auto predictions =
            computeTopKPredictions(
                output.values,
                output.shape,
                class_names,
                top_k
            );

        (void)predictions;
    }

    PipelineBenchmarkResult result;

    result.samples.reserve(
        config.measured_iterations
    );

    for (
        std::size_t iteration = 0;
        iteration < config.measured_iterations;
        ++iteration
    ) {
        // 端到端计时从预处理前开始。
        const auto end_to_end_start =
            PipelineClock::now();

        ImageBatch batch =
            preprocessor.preprocess(image_paths);

        // 预处理结束，同时也是推理开始。
        const auto preprocess_end =
            PipelineClock::now();

        TensorResult output =
            model.run(batch.values, batch.shape);

        // 推理结束，同时也是后处理开始。
        const auto inference_end =
            PipelineClock::now();

        const auto predictions =
            computeTopKPredictions(
                output.values,
                output.shape,
                class_names,
                top_k
            );

        // 后处理和端到端同时结束。
        const auto postprocess_end =
            PipelineClock::now();

        result.samples.push_back(
            PipelineLatencySample{
                durationMilliseconds(
                    end_to_end_start,
                    preprocess_end
                ),
                durationMilliseconds(
                    preprocess_end,
                    inference_end
                ),
                durationMilliseconds(
                    inference_end,
                    postprocess_end
                ),
                durationMilliseconds(
                    end_to_end_start,
                    postprocess_end
                )
            }
        );

        (void)predictions;
    }

    return result;
}


void savePipelineSamplesCsv(
    const std::string& output_path,
    std::size_t batch_size,
    const InferenceBenchmarkConfig& config,
    const PipelineBenchmarkResult& result
) {
    if (result.samples.empty()) {
        throw std::invalid_argument(
            "pipeline result contains no samples"
        );
    }

    const std::filesystem::path csv_path{output_path};

    if (csv_path.has_parent_path()) {
        std::filesystem::create_directories(
            csv_path.parent_path()
        );
    }

    std::ofstream output_file{csv_path};

    if (!output_file.is_open()) {
        throw std::runtime_error(
            "failed to open pipeline CSV: " + output_path
        );
    }

    output_file
        << "iteration,batch_size,warmup_iterations,"
        << "measured_iterations,preprocess_ms,inference_ms,"
        << "postprocess_ms,end_to_end_ms,stage_sum_ms\n"
        << std::setprecision(9);

    for (
        std::size_t index = 0;
        index < result.samples.size();
        ++index
    ) {
        const PipelineLatencySample& sample =
            result.samples[index];

        const double stage_sum_ms =
            sample.preprocess_ms +
            sample.inference_ms +
            sample.postprocess_ms;

        output_file
            << index + 1
            << ','
            << batch_size
            << ','
            << config.warmup_iterations
            << ','
            << config.measured_iterations
            << ','
            << sample.preprocess_ms
            << ','
            << sample.inference_ms
            << ','
            << sample.postprocess_ms
            << ','
            << sample.end_to_end_ms
            << ','
            << stage_sum_ms
            << '\n';
    }

    if (!output_file) {
        throw std::runtime_error(
            "failed while writing pipeline CSV: " + output_path
        );
    }
}
