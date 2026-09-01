#include "inference_benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

// steady_clock只会不断向前走，不受系统时间调整影响，
using BenchmarkClock = std::chrono::steady_clock;

// 输入必须是已经从小到大排好序的延迟数据。
double calculatePercentile(
    const std::vector<double>& sorted_values,
    double percentile
) {
    const double position =
        percentile * static_cast<double>(sorted_values.size() - 1);

    const std::size_t lower_index =
        static_cast<std::size_t>(std::floor(position));

    const std::size_t upper_index =
        static_cast<std::size_t>(std::ceil(position));

    if (lower_index == upper_index) {
        return sorted_values[lower_index];
    }

     const double upper_weight =
        position - static_cast<double>(lower_index);

    const double lower_weight =
        1.0 - upper_weight;

    return
        sorted_values[lower_index] * lower_weight +
        sorted_values[upper_index] * upper_weight;
}

} // namespace


InferenceBenchmarkResult benchmarkInference(
    OnnxModel& model,
    std::vector<float>& input_values,
    const std::vector<std::int64_t>& input_shape,
    const InferenceBenchmarkConfig& config
) {
    // 没有正式测量次数，就无法计算平均值和百分位数。
    if (config.measured_iterations == 0) {
        throw std::invalid_argument(
            "measured_iterations must be positive"
        );
    }

    // input_shape[0]表示batch size。
    if (
        input_shape.empty() ||
        input_shape[0] <= 0
    ) {
        throw std::invalid_argument(
            "input batch size must be positive"
        );
    }

    // 预热阶段正常执行模型，但不记录时间。
    for (
        std::size_t iteration = 0;
        iteration < config.warmup_iterations;
        ++iteration
    ) {
        TensorResult warmup_result =
            model.run(input_values, input_shape);

        (void)warmup_result;
    }

    std::vector<double> latencies_ms;

    latencies_ms.reserve(config.measured_iterations);

    double total_latency_ms = 0.0;

    for(
        std::size_t iteration = 0;
        iteration < config.measured_iterations;
        ++iteration
    ){
        const BenchmarkClock::time_point start_time =
            BenchmarkClock::now();

        TensorResult result =
            model.run(input_values,input_shape);

        const BenchmarkClock::time_point end_time =
            BenchmarkClock::now();

        const double latency_ms =
            std::chrono::duration<double, std::milli>(
                end_time - start_time
            ).count();

        latencies_ms.push_back(latency_ms);
        total_latency_ms += latency_ms;

        (void)result;
    }

    // 记录异常值在原始执行序列中的位置。
    const auto max_latency_iterator =
        std::max_element(
            latencies_ms.begin(),
            latencies_ms.end()
        );

    const std::size_t max_latency_index =
        static_cast<std::size_t>(
            max_latency_iterator - latencies_ms.begin()
        );

    // 排序副本，保留原始延迟的执行顺序。
    std::vector<double> sorted_latencies_ms =
        latencies_ms;

    std::sort(
        sorted_latencies_ms.begin(),
        sorted_latencies_ms.end()
    );

    const double mean_latency_ms =
        total_latency_ms / static_cast<double>(config.measured_iterations);

    const double p50_latency_ms =
        calculatePercentile(sorted_latencies_ms, 0.50);

    const double p95_latency_ms =
        calculatePercentile(sorted_latencies_ms, 0.95);


    // 总图片数量 = 每次推理的图片数 × 推理次数。
    const double processed_images =
        static_cast<double>(input_shape[0]) *
        static_cast<double>(config.measured_iterations);

    // 毫秒转换为秒。
    const double total_time_seconds =
        total_latency_ms / 1000.0;

    const double throughput_images_per_second =
        processed_images / total_time_seconds;

    return InferenceBenchmarkResult{
        mean_latency_ms,
        p50_latency_ms,
        p95_latency_ms,
        sorted_latencies_ms.front(),
        sorted_latencies_ms.back(),
        max_latency_index,
        throughput_images_per_second,
        std::move(latencies_ms)
    };
}


void saveLatencySamplesCsv(
    const std::string& output_path,
    const InferenceBenchmarkConfig& config,
    std::int64_t batch_size,
    const InferenceBenchmarkResult& result
) {
    if (output_path.empty()) {
        throw std::invalid_argument(
            "output CSV path must not be empty"
        );
    }

    const std::filesystem::path csv_path{output_path};

    // 自动创建输出文件的父目录。
    if (csv_path.has_parent_path()) {
        std::filesystem::create_directories(
            csv_path.parent_path()
        );
    }

    std::ofstream output_file{csv_path};

    if (!output_file.is_open()) {
        throw std::runtime_error(
            "failed to open output CSV: " + output_path
        );
    }

    output_file
        << "iteration,batch_size,warmup_iterations,"
        << "measured_iterations,latency_ms,is_max\n"
        << std::setprecision(9);

    for (
        std::size_t index = 0;
        index < result.latencies_ms.size();
        ++index
    ) {
        output_file
            << index + 1
            << ','
            << batch_size
            << ','
            << config.warmup_iterations
            << ','
            << config.measured_iterations
            << ','
            << result.latencies_ms[index]
            << ','
            << (index == result.max_latency_index ? 1 : 0)
            << '\n';
    }

    if (!output_file) {
        throw std::runtime_error(
            "failed while writing output CSV: " + output_path
        );
    }
}


void saveBenchmarkSummaryCsv(
    const std::string& output_path,
    const InferenceBenchmarkConfig& config,
    std::int64_t batch_size,
    const std::vector<InferenceBenchmarkResult>& results
) {
    if (output_path.empty()) {
        throw std::invalid_argument(
            "summary CSV path must not be empty"
        );
    }

    if (results.empty()) {
        throw std::invalid_argument(
            "benchmark summary requires at least one result"
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
            "failed to open summary CSV: " + output_path
        );
    }

    output_file
        << "run,batch_size,warmup_iterations,measured_iterations,"
        << "mean_latency_ms,p50_latency_ms,p95_latency_ms,"
        << "min_latency_ms,max_latency_ms,max_latency_iteration,"
        << "throughput_images_per_second\n"
        << std::setprecision(9);

    for (
        std::size_t run_index = 0;
        run_index < results.size();
        ++run_index
    ) {
        const InferenceBenchmarkResult& result =
            results[run_index];

        output_file
            << run_index + 1
            << ','
            << batch_size
            << ','
            << config.warmup_iterations
            << ','
            << config.measured_iterations
            << ','
            << result.mean_latency_ms
            << ','
            << result.p50_latency_ms
            << ','
            << result.p95_latency_ms
            << ','
            << result.min_latency_ms
            << ','
            << result.max_latency_ms
            << ','
            << result.max_latency_index + 1
            << ','
            << result.throughput_images_per_second
            << '\n';
    }

    if (!output_file) {
        throw std::runtime_error(
            "failed while writing summary CSV: " + output_path
        );
    }
}


void saveTensorValuesCsv(
    const std::string& output_path,
    const TensorResult& result
) {
    if (output_path.empty()) {
        throw std::invalid_argument(
            "tensor output path must not be empty"
        );
    }

    if (result.values.empty()) {
        throw std::invalid_argument(
            "tensor result must not be empty"
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
            "failed to open tensor CSV: " + output_path
        );
    }

    output_file
        << "flat_index,value\n"
        << std::setprecision(9);

    for (
        std::size_t index = 0;
        index < result.values.size();
        ++index
    ) {
        output_file
            << index
            << ','
            << result.values[index]
            << '\n';
    }

    if (!output_file) {
        throw std::runtime_error(
            "failed while writing tensor CSV: " + output_path
        );
    }
}
