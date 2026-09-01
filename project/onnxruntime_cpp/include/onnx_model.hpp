#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

// 保存一次模型推理产生的结果。
struct TensorResult {
    std::vector<std::int64_t> shape;
    std::vector<float> values;
};
// 保存一次ONNX Runtime Session使用的CPU配置。
struct OnnxModelConfig {
    int intra_op_num_threads = 0;
    int inter_op_num_threads = 0;

    ExecutionMode execution_mode = ORT_SEQUENTIAL;

    GraphOptimizationLevel graph_optimization_level =
        ORT_ENABLE_ALL;

    // 性能分析器
    std::string profile_file_prefix;
};


// 负责管理一个ONNX模型及其推理资源。
class OnnxModel {
public:
    // 加载model_path指定的ONNX模型。
    explicit OnnxModel(
        const std::string& model_path,
        const OnnxModelConfig& config = {}
    );

    // 返回模型声明的输入形状，例如[-1, 3, 32, 32]。
    const std::vector<std::int64_t>& inputShape() const noexcept;

    // 使用已有的连续float数据执行一次推理。
    TensorResult run(
        std::vector<float>& input_values,
        const std::vector<std::int64_t>& runtime_input_shape
    );

    // 结束性能记录，并返回实际生成的JSON文件路径。
    std::string endProfiling();

private:
    // 声明顺序非常重要：
    // session依赖environment和session_options，
    // 因此它必须在二者之后声明。
    Ort::Env environment_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;

    Ort::AllocatorWithDefaultOptions allocator_;

    // 复制并保存节点名称，避免依赖临时C字符串的生命周期。
    std::string input_name_;
    std::string output_name_;

    // 保存模型自己的输入契约。
    std::vector<std::int64_t> model_input_shape_;

    // 记录当前Session是否开启了profiling，防止错误调用。
    bool profiling_enabled_;
};
