#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

struct ThreadInfo {
    unsigned int x;
    unsigned int y;
    unsigned int linear_id;
    unsigned int warp_id;
    unsigned int lane_id;
};

__global__ void recordThreadInfo(ThreadInfo* output) {
    // CUDA在Block内先增加x，再增加y。
    const unsigned int linear_id =
        threadIdx.y * blockDim.x + threadIdx.x;

    ThreadInfo info{};
    info.x = threadIdx.x;
    info.y = threadIdx.y;
    info.linear_id = linear_id;
    info.warp_id = linear_id / warpSize;
    info.lane_id = linear_id % warpSize;

    output[linear_id] = info;
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

int main() {
    // dim3按(x, y)书写，因此这里表示16列、16行。
    const dim3 threads_per_block{16, 16};

    const std::size_t thread_count =
        static_cast<std::size_t>(threads_per_block.x) *
        static_cast<std::size_t>(threads_per_block.y);

    const std::size_t byte_count =
        thread_count * sizeof(ThreadInfo);

    std::vector<ThreadInfo> host_info(thread_count);
    ThreadInfo* device_info = nullptr;

    checkCuda(
        cudaMalloc(&device_info, byte_count),
        "cudaMalloc device_info"
    );

    // 只启动一个Block，便于观察Block内部的Warp划分。
    recordThreadInfo<<<1, threads_per_block>>>(device_info);

    checkCuda(
        cudaGetLastError(),
        "recordThreadInfo launch"
    );

    checkCuda(
        cudaDeviceSynchronize(),
        "recordThreadInfo execution"
    );

    checkCuda(
        cudaMemcpy(
            host_info.data(),
            device_info,
            byte_count,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy device_info to host_info"
    );

    checkCuda(
        cudaFree(device_info),
        "cudaFree device_info"
    );

    std::cout
        << "block dimensions: "
        << threads_per_block.x
        << " columns x "
        << threads_per_block.y
        << " rows\n"
        << "thread count: "
        << thread_count
        << '\n'
        << "warp count: "
        << thread_count / 32
        << "\n\n"
        << "linear       x       y    warp    lane\n";

    // 打印前40个线程：完整Warp 0，以及Warp 1的前8个线程。
    const std::size_t displayed_thread_count = 40;

    for (std::size_t index = 0;
         index < displayed_thread_count;
         ++index) {
        const ThreadInfo& info = host_info[index];

        std::cout
            << std::setw(6) << info.linear_id
            << std::setw(8) << info.x
            << std::setw(8) << info.y
            << std::setw(8) << info.warp_id
            << std::setw(8) << info.lane_id
            << '\n';
    }

    return 0;
}
