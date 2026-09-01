#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

constexpr unsigned int kThreadsPerBlock = 256;

// 最大值归约的单位元：任何有效的int64_t都不会比它更小。
constexpr std::int64_t kMaximumIdentity =
    std::numeric_limits<std::int64_t>::lowest();

__global__ void reduceMaxBlocks(
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

    // 越界位置使用最大值运算的单位元，而不是0。
    if (global_index < count) {
        shared_values[thread_id] = input[global_index];
    } else {
        shared_values[thread_id] = kMaximumIdentity;
    }

    // 等待所有真实输入和补充值写入Shared Memory。
    __syncthreads();

    // 每轮让前半部分线程保留左右两个值中的较大者。
    for (unsigned int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2) {
        if (thread_id < stride) {
            const std::int64_t right_value =
                shared_values[thread_id + stride];

            if (right_value > shared_values[thread_id]) {
                shared_values[thread_id] = right_value;
            }
        }

        // 下一轮读取前，必须等待本轮所有比较完成。
        __syncthreads();
    }

    // 每个Block的线程0写出该Block的局部最大值。
    if (thread_id == 0) {
        output[blockIdx.x] = shared_values[0];
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

std::int64_t maximumSerial(
    const std::vector<std::int64_t>& values
) {
    std::int64_t maximum = kMaximumIdentity;

    for (const std::int64_t value : values) {
        if (value > maximum) {
            maximum = value;
        }
    }

    return maximum;
}

unsigned int nextPowerOfTwo(const std::size_t count) {
    unsigned int result = 1;

    while (result < count) {
        result *= 2;
    }

    return result;
}

int main() {
    const std::size_t input_count = 1'000'000;
    std::vector<std::int64_t> host_input(input_count);

    // 使用-1～-7循环，验证越界位置不能错误地补0。
    for (std::size_t index = 0;
         index < input_count;
         ++index) {
        host_input[index] =
            -static_cast<std::int64_t>(index % 7 + 1);
    }

    const std::size_t input_byte_count =
        host_input.size() * sizeof(std::int64_t);

    // 第一轮局部最大值的数量决定两个中间缓冲区的最大容量。
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

        reduceMaxBlocks<<<
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
            "reduceMaxBlocks launch"
        );

        // 本轮每个Block输出一个局部最大值。
        current_count = blocks;
        current_input = current_output;

        // 下一轮切换输出缓冲区，避免覆盖尚未读取的数据。
        current_output =
            current_output == buffer_a
                ? buffer_b
                : buffer_a;
    }

    checkCuda(
        cudaDeviceSynchronize(),
        "all maximum reduction passes execution"
    );

    std::int64_t gpu_maximum = kMaximumIdentity;

    checkCuda(
        cudaMemcpy(
            &gpu_maximum,
            current_input,
            sizeof(std::int64_t),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy final maximum to host"
    );

    checkCuda(cudaFree(device_input), "cudaFree device_input");
    checkCuda(cudaFree(buffer_a), "cudaFree buffer_a");
    checkCuda(cudaFree(buffer_b), "cudaFree buffer_b");

    const std::int64_t cpu_maximum = maximumSerial(host_input);

    std::cout
        << "CPU maximum: "
        << cpu_maximum
        << '\n'
        << "GPU maximum: "
        << gpu_maximum
        << '\n';

    if (gpu_maximum != cpu_maximum) {
        std::cerr << "maximum reduction verification failed\n";
        return 1;
    }

    std::cout << "maximum reduction verification passed\n";

    return 0;
}
