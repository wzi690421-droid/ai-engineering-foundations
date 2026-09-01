#include "onnx_model.hpp"

#include <stdexcept>
#include <cstddef>

namespace {

// 根据实验配置创建SessionOptions。
// 放在匿名namespace中，表示只供当前.cpp使用。
Ort::SessionOptions createSessionOptions(
    const OnnxModelConfig& config
) {
    // 负数没有合理含义。
    if (
        config.intra_op_num_threads < 0 ||
        config.inter_op_num_threads < 0
    ) {
        throw std::invalid_argument(
            "thread counts cannot be negative"
        );
    }

    // 顺序执行模式下设置算子间线程没有实验意义。
    if (
        config.execution_mode == ORT_SEQUENTIAL &&
        config.inter_op_num_threads > 0
    ) {
        throw std::invalid_argument(
            "inter-op threads require ORT_PARALLEL"
        );
    }

    // 创建一份尚未被Session使用的配置对象。
    Ort::SessionOptions options;

    // 大于0时主动设置；0表示保留ORT默认策略。
    if (config.intra_op_num_threads > 0) {
        options.SetIntraOpNumThreads(
            config.intra_op_num_threads
        );
    }

    if (config.inter_op_num_threads > 0) {
        options.SetInterOpNumThreads(
            config.inter_op_num_threads
        );
    }

    // 设置算子顺序/并行执行模式。
    options.SetExecutionMode(
        config.execution_mode
    );

    // 设置图优化级别。
    options.SetGraphOptimizationLevel(
        config.graph_optimization_level
    );

    // profiling必须在Session创建前通过SessionOptions开启。
    if (!config.profile_file_prefix.empty()) {
        options.EnableProfiling(
            config.profile_file_prefix.c_str()
        );
    }

    // 把配置完成的对象交给OnnxModel构造函数。
    return options;
}

}  // namespace

OnnxModel::OnnxModel(
    const std::string& model_path,
    const OnnxModelConfig& config
)
    : environment_{
          ORT_LOGGING_LEVEL_WARNING,
          "onnx_model"
      },

      session_options_{
        createSessionOptions(config)
      },

      session_{
          environment_,
          model_path.c_str(),
          session_options_
      },

      allocator_{},

      profiling_enabled_{
          !config.profile_file_prefix.empty()
      } {

    // 当前这个封装只处理单输入、单输出模型。
    if (
        session_.GetInputCount() != 1 ||
        session_.GetOutputCount() != 1
    ) {
        throw std::runtime_error(
            "OnnxModel currently requires exactly "
            "one input and one output"
        );
    }

    // ONNX Runtime暂时拥有这些字符串。
    auto allocated_input_name =
        session_.GetInputNameAllocated(0, allocator_);

    auto allocated_output_name =
        session_.GetOutputNameAllocated(0, allocator_);

    // 复制到std::string中，由OnnxModel长期拥有。
    input_name_ = allocated_input_name.get();
    output_name_ = allocated_output_name.get();


    const auto input_type_info =
        session_.GetInputTypeInfo(0);


    const auto input_tensor_info =
        input_type_info.GetTensorTypeAndShapeInfo();

    // 当前run()按float缓冲区创建Tensor，
    // 因此模型也必须要求float32输入。
    if (
        input_tensor_info.GetElementType() !=
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    ) {
        throw std::runtime_error(
            "OnnxModel currently supports only float32 input"
        );
    }

    model_input_shape_ = input_tensor_info.GetShape();
}


const std::vector<std::int64_t>& OnnxModel::inputShape() const noexcept {
    return model_input_shape_;
}


// OnnxModel::run表示这是OnnxModel类中的成员函数。
TensorResult OnnxModel::run(
    std::vector<float>& input_values,
    const std::vector<std::int64_t>& runtime_input_shape
) {
    // 实际输入和模型输入必须具有相同的维数。
    // 例如模型是[-1,3,32,32]，实际输入也必须是四维。
    if (
        runtime_input_shape.size() !=
        model_input_shape_.size()
    ) {
        throw std::invalid_argument(
            "runtime input rank does not match model input rank"
        );
    }

    std::size_t expected_element_count = 1;

    for (
        std::size_t index = 0;
        index < runtime_input_shape.size();
        ++index
    ) {
        const std::int64_t runtime_dimension =
            runtime_input_shape[index];

        const std::int64_t model_dimension =
            model_input_shape_[index];

        // 真正运行时不能再包含-1或0。
        if (runtime_dimension <= 0) {
            throw std::invalid_argument(
                "runtime input dimensions must be positive"
            );
        }

        // 模型中大于0的维度是固定维度，必须完全一致。
        // 模型中的-1是动态维度，可以由本次输入决定。
        if (
            model_dimension > 0 &&
            runtime_dimension != model_dimension
        ) {
            throw std::invalid_argument(
                "runtime input shape does not match model input shape"
            );
        }

        expected_element_count *=
            static_cast<std::size_t>(runtime_dimension);
    }

    // 形状要求的元素数量必须和vector实际数量相同。
    if (input_values.size() != expected_element_count) {
        throw std::invalid_argument(
            "input buffer size does not match runtime input shape"
        );
    }

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );

    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(
            memory_info,
            input_values.data(),
            input_values.size(),
            runtime_input_shape.data(),
            runtime_input_shape.size()
        );

    const char* input_names[] = {
        input_name_.c_str()
    };

    const char* output_names[] = {
        output_name_.c_str()
    };

    Ort::RunOptions run_options;

    std::vector<Ort::Value> output_tensors =
        session_.Run(
            run_options,
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

    if (
        output_tensors.size() != 1 ||
        !output_tensors.at(0).IsTensor()
    ) {
        throw std::runtime_error(
            "model did not return exactly one tensor"
        );
    }

    const Ort::Value& output_tensor =
        output_tensors.at(0);

    const auto output_tensor_info =
        output_tensor.GetTensorTypeAndShapeInfo();

    if (
        output_tensor_info.GetElementType() !=
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    ) {
        throw std::runtime_error(
            "OnnxModel currently supports only float32 output"
        );
    }

    TensorResult result;

    result.shape = output_tensor_info.GetShape();

    const std::size_t output_element_count =
        output_tensor_info.GetElementCount();

    const float* output_data =
        output_tensor.GetTensorData<float>();

    result.values.assign(
        output_data,
        output_data + output_element_count
    );

    return result;
}

std::string OnnxModel::endProfiling() {
    // 没有开启profiling，或者已经结束过一次，就不能再次结束。
    if (!profiling_enabled_) {
        throw std::logic_error(
            "profiling is not enabled or has already ended"
        );
    }

    // 通知ORT停止记录并写出JSON文件。
    // 返回对象负责暂时管理ORT分配的路径字符串。
    auto allocated_profile_path =
        session_.EndProfilingAllocated(allocator_);

    // 在临时对象销毁之前，把路径复制到自己的std::string中。
    const std::string profile_path{
        allocated_profile_path.get()
    };

    profiling_enabled_ = false;

    return profile_path;
}
