#include "image_preprocessor.hpp"
#include "inference_benchmark.hpp"
#include "onnx_model.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// 完整解析命令行中的迭代次数，拒绝abc、3abc和负数。
std::size_t parseIterationCount(
    const std::string& text,
    const std::string& argument_name,
    bool allow_zero
) {
    std::size_t parsed_character_count = 0;
    long long parsed_value = 0;

    try {
        parsed_value = std::stoll(
            text,
            &parsed_character_count
        );
    } catch (const std::exception&) {
        throw std::invalid_argument(
            argument_name + " must be a complete integer"
        );
    }

    if (parsed_character_count != text.size()) {
        throw std::invalid_argument(
            argument_name + " must be a complete integer"
        );
    }

    if (
        parsed_value < 0 ||
        (!allow_zero && parsed_value == 0)
    ) {
        throw std::invalid_argument(
            argument_name +
            (allow_zero ? " must be non-negative" : " must be positive")
        );
    }

    const auto unsigned_value =
        static_cast<unsigned long long>(parsed_value);

    if (
        unsigned_value >
        static_cast<unsigned long long>(
            std::numeric_limits<std::size_t>::max()
        )
    ) {
        throw std::out_of_range(
            argument_name + " is too large"
        );
    }

    return static_cast<std::size_t>(unsigned_value);
}


// 把命令行文本转换为ORT的图优化枚举。
GraphOptimizationLevel parseGraphOptimizationLevel(
    const std::string& text
) {
    if (text == "disabled") {
        return ORT_DISABLE_ALL;
    }

    if (text == "basic") {
        return ORT_ENABLE_BASIC;
    }

    if (text == "extended") {
        return ORT_ENABLE_EXTENDED;
    }

    if (text == "all") {
        return ORT_ENABLE_ALL;
    }

    throw std::invalid_argument(
        "graph optimization must be "
        "disabled, basic, extended, or all"
    );
}

}  // namespace


int main(int argc, char* argv[]) {
    // argv[1]：模型；argv[2]：配置；argv[3]：预热次数；
    // argv[4]：测量次数；argv[5]：重复轮数；
    // argv[6]：结果目录；argv[7]：intra-op线程数；
    // argv[8]：图优化级别；argv[9]：是否开启profiling；
    // argv[10]以后：图片。
    if (argc < 11) {
        std::cerr
            << "usage: "
            << argv[0]
            << " <model_path> <config_path>"
            << " <warmup_iterations> <measured_iterations>"
            << " <repeat_count> <output_directory>"
            << " <intra_op_threads>"
            << " <graph_optimization>"
            << " <profiling: off|profile>"
            << " <image_path> [image_path ...]\n";

        return 1;
    }

    try {
        const InferenceBenchmarkConfig benchmark_config{
            parseIterationCount(argv[3], "warmup_iterations", true),
            parseIterationCount(argv[4], "measured_iterations", false)
        };

        const std::size_t repeat_count =
            parseIterationCount(argv[5], "repeat_count", false);

        const std::filesystem::path output_directory{argv[6]};

        // profiling结束时需要向这里写JSON文件，
        // 因此在创建Session之前保证目录存在。
        std::filesystem::create_directories(
            output_directory
        );

        // 先使用已有函数，把线程数文本转换为size_t。
        const std::size_t parsed_intra_op_threads =
            parseIterationCount(
                argv[7],
                "intra_op_threads",
                true
            );

        if (
            parsed_intra_op_threads >
            static_cast<std::size_t>(
                std::numeric_limits<int>::max()
            )
        ) {
            throw std::out_of_range(
                "intra_op_threads is too large"
            );
        }

        // 创建模型配置，未修改的字段继续使用结构体默认值。
        OnnxModelConfig model_config;

        // size_t已经确认不会超过int范围，可以安全转换。
        model_config.intra_op_num_threads =
            static_cast<int>(parsed_intra_op_threads);

        // 把disabled/basic/extended/all转换成ORT枚举。
        model_config.graph_optimization_level =
            parseGraphOptimizationLevel(argv[8]);

        const std::string profiling_mode{argv[9]};

        if (profiling_mode == "profile") {
            // 这里只设置文件名前缀；
            // ORT会自动添加时间戳和.json后缀。
            model_config.profile_file_prefix =
                (
                    output_directory /
                    "ort_profile"
                ).string();
        } else if (profiling_mode != "off") {
            throw std::invalid_argument(
                "profiling must be off or profile"
            );
        }

        std::vector<std::string> image_paths;
        image_paths.reserve(
            static_cast<std::size_t>(argc - 10)
        );

        for (
            int argument_index = 10;
            argument_index < argc;
            ++argument_index
        ) {
            image_paths.emplace_back(argv[argument_index]);
        }

        const NormalizationParameters normalization =
            loadNormalizationParameters(argv[2]);

        // 使用本次命令行指定的CPU配置创建Session。
        OnnxModel model{
            argv[1],
            model_config
        };

        const std::vector<std::int64_t>& model_input_shape =
            model.inputShape();

        if (
            model_input_shape.size() != 4 ||
            model_input_shape[1] != 3 ||
            model_input_shape[2] <= 0 ||
            model_input_shape[3] <= 0
        ) {
            throw std::runtime_error(
                "expected model input shape [batch, 3, height, width]"
            );
        }

        ImagePreprocessor preprocessor{
            static_cast<int>(model_input_shape[2]),
            static_cast<int>(model_input_shape[3]),
            normalization
        };

        // 图片只读取和预处理一次；正式计时只测model.run()。
        ImageBatch batch =
            preprocessor.preprocess(image_paths);

        // 在正式benchmark前额外推理一次并保存全部输出。
        // 这次调用不属于后面的延迟统计。
        TensorResult verification_output =
            model.run(
                batch.values,
                batch.shape
            );

        const std::filesystem::path logits_csv_path =
            output_directory / "logits.csv";

        saveTensorValuesCsv(
            logits_csv_path.string(),
            verification_output
        );

        std::cout
            << std::fixed
            << std::setprecision(6);

        std::vector<InferenceBenchmarkResult> results;
        results.reserve(repeat_count);

        for (
            std::size_t run_index = 0;
            run_index < repeat_count;
            ++run_index
        ) {
            InferenceBenchmarkResult result =
                benchmarkInference(
                    model,
                    batch.values,
                    batch.shape,
                    benchmark_config
                );

            const std::filesystem::path raw_csv_path =
                output_directory /
                (
                    "run_" +
                    std::to_string(run_index + 1) +
                    "_raw.csv"
                );

            saveLatencySamplesCsv(
                raw_csv_path.string(),
                benchmark_config,
                batch.shape[0],
                result
            );

            std::cout
                << "run: "
                << run_index + 1
                << '\n'
                << "batch_size: "
                << batch.shape[0]
                << '\n'
                << "mean_latency_ms: "
                << result.mean_latency_ms
                << '\n'
                << "p50_latency_ms: "
                << result.p50_latency_ms
                << '\n'
                << "p95_latency_ms: "
                << result.p95_latency_ms
                << '\n'
                << "max_latency_ms: "
                << result.max_latency_ms
                << '\n'
                << "throughput_images_per_second: "
                << result.throughput_images_per_second
                << "\n\n";

            results.push_back(std::move(result));
        }

        const std::filesystem::path summary_csv_path =
            output_directory / "summary.csv";

        saveBenchmarkSummaryCsv(
            summary_csv_path.string(),
            benchmark_config,
            batch.shape[0],
            results
        );

        std::cout
            << "repeat_count: "
            << repeat_count
            << '\n'
            << "batch_size: "
            << batch.shape[0]
            << '\n'
            << "warmup_iterations: "
            << benchmark_config.warmup_iterations
            << '\n'
            << "measured_iterations: "
            << benchmark_config.measured_iterations
            << '\n'
            << "summary_csv: "
            << summary_csv_path.string()
            << '\n';

        if (profiling_mode == "profile") {
            // 停止记录、写出JSON，并取得真实文件路径。
            const std::string profile_json_path =
                model.endProfiling();

            std::cout
                << "profile_json: "
                << profile_json_path
                << '\n';
        }

        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "error: "
            << exception.what()
            << '\n';

        return 1;
    }
}
