#include <cuda_runtime.h>

#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <vector>

// Kernel：由CPU启动，在GPU上执行。
__global__ void addOne(int* values, const std::size_t count) {
    // 每个CUDA线程计算自己的全局下标。
    const std::size_t index =
        blockIdx.x * blockDim.x + threadIdx.x;

    // 多余线程不能访问数组。
    if (index < count) {
        values[index] += 1;
    }
}

void checkCuda(
    const cudaError_t status,
    const char* operation
) {
    // cudaSuccess表示操作成功。
    if (status != cudaSuccess) {
    std::cerr
        << operation
        << " failed: "
        << cudaGetErrorString(status)
        << '\n';

    // 发生CUDA错误后立即结束程序。
    std::exit(EXIT_FAILURE);
    }
}

int main() {
    // 数据首先存在CPU内存中。
    std::vector<int> host_values{3, 1, 4, 1, 5};

    const std::size_t count = host_values.size();
    const std::size_t byte_count = count * sizeof(int);

    // 用于保存GPU显存地址。
    int* device_values = nullptr;

    // 在GPU上申请显存。
    checkCuda(
        cudaMalloc(&device_values, byte_count),
        "cudaMalloc"
    );

    // 把输入从CPU复制到GPU。
    checkCuda(
        cudaMemcpy(
            device_values,
            host_values.data(),
            byte_count,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy HostToDevice"
    );

    // 决定GPU线程的组织方式。
    const std::size_t threads_per_block = 256;
    const std::size_t block_count =
        (count + threads_per_block - 1) / threads_per_block;

    // 启动GPU Kernel。
    addOne<<<block_count, threads_per_block>>>(
        device_values,
        count
    );

    // 检查Kernel启动参数和配置。
    checkCuda(
        cudaGetLastError(),
        "addOne launch"
    );

    // 等待GPU执行结束，并检查执行期间的错误。
    checkCuda(
        cudaDeviceSynchronize(),
        "addOne execution"
    );

    // 把结果从GPU复制回CPU。
    checkCuda(
        cudaMemcpy(
            host_values.data(),
            device_values,
            byte_count,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy DeviceToHost"
    );

    // 释放GPU显存。
    checkCuda(
        cudaFree(device_values),
        "cudaFree"
    );

    std::cout << "result:";

    for (const int value : host_values) {
        std::cout << ' ' << value;
    }

    std::cout << '\n';

    return 0;
}
