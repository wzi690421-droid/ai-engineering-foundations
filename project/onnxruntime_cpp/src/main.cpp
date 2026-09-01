#include "classification_postprocessor.hpp"
#include "image_preprocessor.hpp"
#include "onnx_model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>


// 打印Tensor形状，例如[1, 3, 32, 32]。
void printShape(
    const std::vector<std::int64_t>& shape
) {
    std::cout << '[';

    for (
        std::size_t index = 0;
        index < shape.size();
        ++index
    ) {
        if (index != 0) {
            std::cout << ", ";
        }

        std::cout << shape[index];
    }

    std::cout << ']';
}

// 打印RGB三个通道的参数，例如mean或std。
void printThreeValues(
    // std::array<float,3>长度固定为3。
    const std::array<float, 3>& values
) {
    std::cout << '[';

    for (
        std::size_t index = 0;
        index < values.size();
        ++index
    ) {
        if (index != 0) {
            std::cout << ", ";
        }

        std::cout << values[index];
    }

    std::cout << ']';
}


// 把命令行中的Top-K文本转换为正整数。
std::size_t parseTopK(const std::string& text) {
    // std::stoll会从字符串开头读取整数，并把实际读取的字符数
    // 写入parsed_character_count。
    std::size_t parsed_character_count = 0;
    long long parsed_value = 0;

    try {
        parsed_value = std::stoll(
            text,
            &parsed_character_count
        );
    } catch (const std::exception&) {
        throw std::invalid_argument(
            "top_k must be a complete positive integer"
        );
    }

    // 已读取的字符数是否等于整个字符串长度。
    if (
        parsed_character_count != text.size() ||
        parsed_value <= 0
    ) {
        throw std::invalid_argument(
            "top_k must be a complete positive integer"
        );
    }

    // 防止在size_t比long long更窄的平台上发生整数截断。
    const auto unsigned_value =
        static_cast<unsigned long long>(parsed_value);

    if (
        unsigned_value >
        static_cast<unsigned long long>(
            std::numeric_limits<std::size_t>::max()
        )
    ) {
        throw std::out_of_range(
            "top_k is too large"
        );
    }

    return static_cast<std::size_t>(unsigned_value);
}


int main(int argc, char* argv[]) {
    // argv[0]：程序路径；
    // argv[1]：ONNX模型路径；
    // argv[2]：config.json路径；
    // argv[3]：需要输出的预测数量Top-K；
    // argv[4]及以后：一张或多张图片路径。
    if (argc < 5) {
        std::cerr
            << "usage: "
            << argv[0]
            << " <model_path>"
            << " <config_path>"
            << " <top_k>"
            << " <image_path> [image_path ...]\n";

        return 1;
    }

    try {
        const std::size_t top_k = parseTopK(argv[3]);

        std::vector<std::string> image_paths;

        image_paths.reserve(
            static_cast<std::size_t>(argc - 4)
        );

        for (
            int argument_index = 4;
            argument_index < argc;
            ++argument_index
        ) {
            image_paths.emplace_back(
                argv[argument_index]
            );
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
                "expected model input shape "
                "[batch, 3, height, width]"
            );
        }

        // 模型形状排列是[N,C,H,W]：
        const int target_height =
            static_cast<int>(model_input_shape[2]);

        const int target_width =
            static_cast<int>(model_input_shape[3]);

        // 创建图片预处理器。
        ImagePreprocessor preprocessor{
            target_height,
            target_width,
            normalization
        };

        // 读取并处理全部图片。
        ImageBatch batch =
            preprocessor.preprocess(image_paths);

        // 把预处理后的连续float和shape交给ONNX模型。
        TensorResult result =
            model.run(
                batch.values,
                batch.shape
            );

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

        // 对每张图片计算稳定Softmax并保留用户要求的Top-K。
        const std::vector<std::vector<ClassPrediction>> predictions =
            computeTopKPredictions(
                result.values,
                result.shape,
                class_names,
                top_k
            );

        // 打印从config.json读到的参数，
        std::cout << "mean: ";
        printThreeValues(normalization.mean);
        std::cout << '\n';

        std::cout << "std: ";
        printThreeValues(normalization.std);
        std::cout << '\n';

        std::cout << "runtime input shape: ";
        printShape(batch.shape);
        std::cout << '\n';

        std::cout << "runtime output shape: ";
        printShape(result.shape);
        std::cout << '\n';

        // 分别打印每张图片的Top-K预测。
        for (
            std::size_t image_index = 0;
            image_index < image_paths.size();
            ++image_index
        ) {
            std::cout
                << "image: "
                << image_paths[image_index]
                << '\n';

            const std::vector<ClassPrediction>& sample_predictions =
                predictions.at(image_index);

            for (
                std::size_t rank = 0;
                rank < sample_predictions.size();
                ++rank
            ) {
                const ClassPrediction& prediction =
                    sample_predictions[rank];

                std::cout
                    << "  "
                    << rank + 1
                    << ". "
                    << prediction.class_name
                    << " (class "
                    << prediction.class_index
                    << "): "
                    << std::fixed
                    << std::setprecision(2)
                    << prediction.probability * 100.0F
                    << "%"
                    << std::defaultfloat
                    << std::setprecision(6)
                    << ", logit="
                    << prediction.logit
                    << '\n';
            }
        }

        return 0;

    // std::exception可以接住错误,然后交给catch处理
    } catch (const std::exception& exception) {
        std::cerr
            << "error: "
            << exception.what()
            << '\n';

        return 1;
    }
}
