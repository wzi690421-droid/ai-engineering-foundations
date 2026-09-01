#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

constexpr unsigned int kThreadsPerBlock = 256;

// 基线版：每一轮都通过Shared Memory交换中间结果。
__global__ void reduceBlocksShared(
    const std::int64_t* input,
    std::int64_t* output,
    const std::size_t count
) {
    // 大小由Kernel启动时的第三个配置参数决定。
    extern __shared__ std::int64_t shared_values[];

    const unsigned int thread_id = threadIdx.x;
    const std::size_t global_index =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        thread_id;

    // 没有对应输入的线程写入0，使任意长度都能按2的幂归约。
    if (global_index < count) {
        shared_values[thread_id] = input[global_index];
    } else {
        shared_values[thread_id] = 0;
    }

    // 等待所有真实输入和补零位置全部写入Shared Memory。
    __syncthreads();

    // stride从当前Block线程数的一半开始，每轮继续减半。
    for (unsigned int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2) {
        if (thread_id < stride) {
            shared_values[thread_id] +=
                shared_values[thread_id + stride];
        }

        // 下一轮读取数据前，必须等待本轮所有加法完成。
        __syncthreads();
    }

    // 每个Block的线程0写出该Block的部分和。
    if (thread_id == 0) {
        output[blockIdx.x] = shared_values[0];
    }
}

// 优化版：多个Warp之间使用Shared Memory，最后一个Warp使用Shuffle。
__global__ void reduceBlocksWarpShuffle(
    const std::int64_t* input,
    std::int64_t* output,
    const std::size_t count
) {
    extern __shared__ std::int64_t shared_values[];

    const unsigned int thread_id = threadIdx.x;
    const std::size_t global_index =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        thread_id;

    if (global_index < count) {
        shared_values[thread_id] = input[global_index];
    } else {
        shared_values[thread_id] = 0;
    }

    __syncthreads();

    // 对256线程的Block，这里只执行stride=128、64、32。
    for (unsigned int stride = blockDim.x / 2;
         stride >= warpSize;
         stride /= 2) {
        if (thread_id < stride) {
            shared_values[thread_id] +=
                shared_values[thread_id + stride];
        }

        __syncthreads();
    }

    if (thread_id < warpSize) {
        // 正常Block有32个部分和；最后一趟可能只有更少的有效Lane。
        std::int64_t value = shared_values[thread_id];
        const unsigned int active_mask = __activemask();
        const unsigned int active_lane_count =
            blockDim.x < warpSize
                ? blockDim.x
                : warpSize;

        // 当前主流程保证线程数是2的幂，因此可以从有效Lane数的一半开始。
        for (unsigned int offset = active_lane_count / 2;
             offset > 0;
             offset /= 2) {
            // 所有活跃Lane都执行Shuffle，前半部分Lane使用读取结果。
            const std::int64_t other_value =
                __shfl_down_sync(
                    active_mask,
                    value,
                    offset
                );

            if (thread_id < offset) {
                value += other_value;
            }
        }

        if (thread_id == 0) {
            output[blockIdx.x] = value;
        }
    }
}

void checkCuda(
    const cudaError_t status,
    const char* operation
) {
    if (status != cudaSuccess) {
        std::cerr
            << operation
            << " failed: "
            << cudaGetErrorString(status)
            << '\n';

        std::exit(EXIT_FAILURE);
    }
}

std::int64_t sumSerial(
    const std::vector<std::int64_t>& values
) {
    std::int64_t sum = 0;

    for (const std::int64_t value : values) {
        sum += value;
    }

    return sum;
}

unsigned int nextPowerOfTwo(const std::size_t count) {
    unsigned int result = 1;

    while (result < count) {
        result *= 2;
    }

    return result;
}

void launchReductionKernel(
    const bool use_warp_shuffle,
    const unsigned int blocks,
    const unsigned int threads,
    const std::size_t shared_memory_byte_count,
    const std::int64_t* input,
    std::int64_t* output,
    const std::size_t count
) {
    if (use_warp_shuffle) {
        reduceBlocksWarpShuffle<<<
            blocks,
            threads,
            shared_memory_byte_count
        >>>(
            input,
            output,
            count
        );
    } else {
        reduceBlocksShared<<<
            blocks,
            threads,
            shared_memory_byte_count
        >>>(
            input,
            output,
            count
        );
    }
}

float measureFirstPassMilliseconds(
    const bool use_warp_shuffle,
    const unsigned int iterations,
    const unsigned int blocks,
    const unsigned int threads,
    const std::size_t shared_memory_byte_count,
    const std::int64_t* input,
    std::int64_t* output,
    const std::size_t count
) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    checkCuda(cudaEventRecord(start), "cudaEventRecord start");

    for (unsigned int iteration = 0;
         iteration < iterations;
         ++iteration) {
        launchReductionKernel(
            use_warp_shuffle,
            blocks,
            threads,
            shared_memory_byte_count,
            input,
            output,
            count
        );
    }

    checkCuda(
        cudaGetLastError(),
        use_warp_shuffle
            ? "reduceBlocksWarpShuffle benchmark launch"
            : "reduceBlocksShared benchmark launch"
    );
    checkCuda(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float total_milliseconds = 0.0F;
    checkCuda(
        cudaEventElapsedTime(
            &total_milliseconds,
            start,
            stop
        ),
        "cudaEventElapsedTime"
    );

    checkCuda(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy stop");

    return total_milliseconds /
        static_cast<float>(iterations);
}

int main() {
    const std::size_t input_count = 1'000'000;
    std::vector<std::int64_t> host_input(input_count);

    // 使用1～7循环，既避免结果过于单一，也不会产生溢出。
    for (std::size_t index = 0;
         index < input_count;
         ++index) {
        host_input[index] =
            static_cast<std::int64_t>(index % 7 + 1);
    }

    const std::size_t input_byte_count =
        host_input.size() * sizeof(std::int64_t);

    // 第一轮产生的部分和数量，是后续两个缓冲区所需的最大容量。
    const std::size_t intermediate_capacity =
        (input_count + kThreadsPerBlock - 1) /
        kThreadsPerBlock;

    const std::size_t intermediate_byte_count =
        intermediate_capacity * sizeof(std::int64_t);

    std::int64_t* device_input = nullptr;
    std::int64_t* buffer_a = nullptr;
    std::int64_t* buffer_b = nullptr;

    checkCuda(
        cudaMalloc(&device_input, input_byte_count),
        "cudaMalloc device_input"
    );
    checkCuda(
        cudaMalloc(&buffer_a, intermediate_byte_count),
        "cudaMalloc buffer_a"
    );
    checkCuda(
        cudaMalloc(&buffer_b, intermediate_byte_count),
        "cudaMalloc buffer_b"
    );

    checkCuda(
        cudaMemcpy(
            device_input,
            host_input.data(),
            input_byte_count,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy host_input to device_input"
    );

    constexpr unsigned int warmup_iterations = 20;
    constexpr unsigned int benchmark_iterations = 1'000;
    constexpr unsigned int benchmark_rounds = 5;

    const unsigned int first_pass_threads = kThreadsPerBlock;
    const unsigned int first_pass_blocks =
        static_cast<unsigned int>(
            (input_count + first_pass_threads - 1) /
            first_pass_threads
        );
    const std::size_t first_pass_shared_memory_byte_count =
        static_cast<std::size_t>(first_pass_threads) *
        sizeof(std::int64_t);

    // 两个版本交替预热，排除首次运行和单一版本先后顺序的影响。
    for (unsigned int iteration = 0;
         iteration < warmup_iterations;
         ++iteration) {
        launchReductionKernel(
            false,
            first_pass_blocks,
            first_pass_threads,
            first_pass_shared_memory_byte_count,
            device_input,
            buffer_a,
            input_count
        );
        launchReductionKernel(
            true,
            first_pass_blocks,
            first_pass_threads,
            first_pass_shared_memory_byte_count,
            device_input,
            buffer_a,
            input_count
        );
    }

    checkCuda(cudaGetLastError(), "reduction benchmark warmup launch");
    checkCuda(cudaDeviceSynchronize(), "reduction benchmark warmup");

    double shared_total_milliseconds = 0.0;
    double shuffle_total_milliseconds = 0.0;

    // 每轮交换测试顺序，降低温度和GPU频率变化造成的顺序偏差。
    for (unsigned int round = 0;
         round < benchmark_rounds;
         ++round) {
        if (round % 2 == 0) {
            shared_total_milliseconds += measureFirstPassMilliseconds(
                false,
                benchmark_iterations,
                first_pass_blocks,
                first_pass_threads,
                first_pass_shared_memory_byte_count,
                device_input,
                buffer_a,
                input_count
            );
            shuffle_total_milliseconds += measureFirstPassMilliseconds(
                true,
                benchmark_iterations,
                first_pass_blocks,
                first_pass_threads,
                first_pass_shared_memory_byte_count,
                device_input,
                buffer_a,
                input_count
            );
        } else {
            shuffle_total_milliseconds += measureFirstPassMilliseconds(
                true,
                benchmark_iterations,
                first_pass_blocks,
                first_pass_threads,
                first_pass_shared_memory_byte_count,
                device_input,
                buffer_a,
                input_count
            );
            shared_total_milliseconds += measureFirstPassMilliseconds(
                false,
                benchmark_iterations,
                first_pass_blocks,
                first_pass_threads,
                first_pass_shared_memory_byte_count,
                device_input,
                buffer_a,
                input_count
            );
        }
    }

    const double shared_average_milliseconds =
        shared_total_milliseconds /
        static_cast<double>(benchmark_rounds);
    const double shuffle_average_milliseconds =
        shuffle_total_milliseconds /
        static_cast<double>(benchmark_rounds);

    std::cout
        << std::fixed
        << std::setprecision(3)
        << "first-pass Shared Memory average: "
        << shared_average_milliseconds * 1'000.0
        << " us\n"
        << "first-pass Warp Shuffle average: "
        << shuffle_average_milliseconds * 1'000.0
        << " us\n"
        << "first-pass speedup: "
        << shared_average_milliseconds /
            shuffle_average_milliseconds
        << "x\n";

    const std::int64_t* current_input = device_input;
    std::int64_t* current_output = buffer_a;
    std::size_t current_count = input_count;
    unsigned int pass = 0;

    std::cout
        << "input count: "
        << input_count
        << '\n'
        << "intermediate buffer capacity: "
        << intermediate_capacity
        << '\n';

    while (current_count > 1) {
        ++pass;

        const unsigned int threads =
            current_count < kThreadsPerBlock
                ? nextPowerOfTwo(current_count)
                : kThreadsPerBlock;

        const unsigned int blocks =
            static_cast<unsigned int>(
                (current_count + threads - 1) /
                threads
            );

        const std::size_t shared_memory_byte_count =
            static_cast<std::size_t>(threads) *
            sizeof(std::int64_t);

        std::cout
            << "pass "
            << pass
            << ": input="
            << current_count
            << ", threads="
            << threads
            << ", blocks="
            << blocks
            << ", output="
            << blocks
            << '\n';

        reduceBlocksWarpShuffle<<<
            blocks,
            threads,
            shared_memory_byte_count
        >>>(
            current_input,
            current_output,
            current_count
        );

        checkCuda(
            cudaGetLastError(),
            "reduceBlocksWarpShuffle launch"
        );

        // 本轮每个Block输出一个部分和。
        current_count = blocks;
        current_input = current_output;

        // 下一轮使用另一个缓冲区作为输出，避免覆盖尚未读取的数据。
        current_output =
            current_output == buffer_a
                ? buffer_b
                : buffer_a;
    }

    checkCuda(
        cudaDeviceSynchronize(),
        "all reduction passes execution"
    );

    std::int64_t gpu_sum = 0;

    checkCuda(
        cudaMemcpy(
            &gpu_sum,
            current_input,
            sizeof(std::int64_t),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy final sum to host"
    );

    checkCuda(cudaFree(device_input), "cudaFree device_input");
    checkCuda(cudaFree(buffer_a), "cudaFree buffer_a");
    checkCuda(cudaFree(buffer_b), "cudaFree buffer_b");

    const std::int64_t cpu_sum = sumSerial(host_input);

    std::cout
        << "CPU sum: "
        << cpu_sum
        << '\n'
        << "GPU sum: "
        << gpu_sum
        << '\n';

    if (gpu_sum != cpu_sum) {
        std::cerr << "reduction verification failed\n";
        return 1;
    }

    std::cout << "reduction verification passed\n";

    return 0;
}
