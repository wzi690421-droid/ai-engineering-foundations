#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

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
    // 0表示系统中的第一张CUDA GPU。
    const int device_index = 0;

    // 用于接收GPU硬件信息。
    cudaDeviceProp properties{};

    // 查询第0张GPU的属性。
    checkCuda(
        cudaGetDeviceProperties(
            &properties,
            device_index
        ),
        "cudaGetDeviceProperties"
    );

    std::cout
        << "device name: "
        << properties.name
        << '\n'

        << "compute capability: "
        << properties.major
        << '.'
        << properties.minor
        << '\n'

        << "SM count: "
        << properties.multiProcessorCount
        << '\n'

        << "warp size: "
        << properties.warpSize
        << '\n'

        << "maximum threads per block: "
        << properties.maxThreadsPerBlock
        << '\n'

        << "maximum threads per SM: "
        << properties.maxThreadsPerMultiProcessor
        << '\n'

        << "maximum blocks per SM: "
        << properties.maxBlocksPerMultiProcessor
        << '\n'

        << "maximum block dimensions: ["
        << properties.maxThreadsDim[0]
        << ", "
        << properties.maxThreadsDim[1]
        << ", "
        << properties.maxThreadsDim[2]
        << "]\n"

        << "maximum grid dimensions: ["
        << properties.maxGridSize[0]
        << ", "
        << properties.maxGridSize[1]
        << ", "
        << properties.maxGridSize[2]
        << "]\n";

    return 0;
}
