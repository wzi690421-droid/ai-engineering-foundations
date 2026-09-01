#include "image_preprocessor.hpp"
#include "inference_benchmark.hpp"
#include "onnx_model.hpp"
#include "pipeline_benchmark.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::size_t parsePositiveInteger(
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

    if (
        parsed_character_count != text.size() ||
        parsed_value < 0 ||
        (!allow_zero && parsed_value == 0)
    ) {
        throw std::invalid_argument(
            argument_name + " has an invalid value"
        );
    }

    return static_cast<std::size_t>(parsed_value);
}

}  // namespace


int main(int argc, char* argv[]) {
    // 模型、配置、预热次数、测量次数、CSV、Top-K、图片。
    if (argc < 8) {
        std::cerr
            << "usage: "
            << argv[0]
            << " <model_path> <config_path>"
            << " <warmup_iterations> <measured_iterations>"
            << " <output_csv_path> <top_k>"
            << " <image_path> [image_path ...]\n";

        return 1;
    }

    try {
        const InferenceBenchmarkConfig config{
            parsePositiveInteger(argv[3], "warmup_iterations", true),
            parsePositiveInteger(argv[4], "measured_iterations", false)
        };

        const std::size_t top_k =
            parsePositiveInteger(argv[6], "top_k", false);

        std::vector<std::string> image_paths;
        image_paths.reserve(
            static_cast<std::size_t>(argc - 7)
        );

        for (
            int argument_index = 7;
            argument_index < argc;
            ++argument_index
        ) {
            image_paths.emplace_back(argv[argument_index]);
        }

        const NormalizationParameters normalization =
            loadNormalizationParameters(argv[2]);

        OnnxModel model{argv[1]};

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

        const std::vector<std::string> class_names{
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
        };

        const PipelineBenchmarkResult result =
            benchmarkPipeline(
                model,
                preprocessor,
                image_paths,
                class_names,
                top_k,
                config
            );

        savePipelineSamplesCsv(
            argv[5],
            image_paths.size(),
            config,
            result
        );

        double preprocess_total_ms = 0.0;
        double inference_total_ms = 0.0;
        double postprocess_total_ms = 0.0;
        double end_to_end_total_ms = 0.0;

        for (const PipelineLatencySample& sample : result.samples) {
            preprocess_total_ms += sample.preprocess_ms;
            inference_total_ms += sample.inference_ms;
            postprocess_total_ms += sample.postprocess_ms;
            end_to_end_total_ms += sample.end_to_end_ms;
        }

        const double sample_count =
            static_cast<double>(result.samples.size());

        const double preprocess_mean_ms =
            preprocess_total_ms / sample_count;
        const double inference_mean_ms =
            inference_total_ms / sample_count;
        const double postprocess_mean_ms =
            postprocess_total_ms / sample_count;
        const double end_to_end_mean_ms =
            end_to_end_total_ms / sample_count;

        std::cout
            << std::fixed
            << std::setprecision(6)
            << "batch_size: "
            << image_paths.size()
            << '\n'
            << "preprocess_mean_ms: "
            << preprocess_mean_ms
            << '\n'
            << "inference_mean_ms: "
            << inference_mean_ms
            << '\n'
            << "postprocess_mean_ms: "
            << postprocess_mean_ms
            << '\n'
            << "end_to_end_mean_ms: "
            << end_to_end_mean_ms
            << '\n'
            << "preprocess_share_percent: "
            << preprocess_mean_ms / end_to_end_mean_ms * 100.0
            << '\n'
            << "inference_share_percent: "
            << inference_mean_ms / end_to_end_mean_ms * 100.0
            << '\n'
            << "postprocess_share_percent: "
            << postprocess_mean_ms / end_to_end_mean_ms * 100.0
            << '\n'
            << "pipeline_csv: "
            << argv[5]
            << '\n';

        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "error: "
            << exception.what()
            << '\n';

        return 1;
    }
}
