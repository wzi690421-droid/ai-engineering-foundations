#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

enum class AccessPattern {
    coalesced,
    strided
};

__global__ void matrixAddCoalesced(
    const int* a,
    const int* b,
    int* c,
    const std::size_t row_count,
    const std::size_t column_count
) {
    // 同一个Warp的x连续变化，因此column连续变化。
    const std::size_t column =
        blockIdx.x * blockDim.x + threadIdx.x;

    const std::size_t row =
        blockIdx.y * blockDim.y + threadIdx.y;

    if (row < row_count && column < column_count) {
        const std::size_t index =
            row * column_count + column;

        c[index] = a[index] + b[index];
    }
}

__global__ void matrixAddStrided(
    const int* a,
    const int* b,
    int* c,
    const std::size_t row_count,
    const std::size_t column_count
) {
    // 列主序下标使相邻线程跨越row_count个int。
    const std::size_t column =
        blockIdx.x * blockDim.x + threadIdx.x;

    const std::size_t row =
        blockIdx.y * blockDim.y + threadIdx.y;

    if (row < row_count && column < column_count) {
        const std::size_t index =
            column * row_count + row;

        c[index] = a[index] + b[index];
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

void launchMatrixAdd(
    const AccessPattern pattern,
    const dim3 block_count,
    const dim3 threads_per_block,
    const int* device_a,
    const int* device_b,
    int* device_c,
    const std::size_t row_count,
    const std::size_t column_count
) {
    if (pattern == AccessPattern::coalesced) {
        matrixAddCoalesced<<<block_count, threads_per_block>>>(
            device_a,
            device_b,
            device_c,
            row_count,
            column_count
        );
    } else {
        matrixAddStrided<<<block_count, threads_per_block>>>(
            device_a,
            device_b,
            device_c,
            row_count,
            column_count
        );
    }
}

double benchmarkMatrixAdd(
    const AccessPattern pattern,
    const dim3 block_count,
    const dim3 threads_per_block,
    const int* device_a,
    const int* device_b,
    int* device_c,
    const std::size_t row_count,
    const std::size_t column_count,
    const int warmup_iterations,
    const int measured_iterations
) {
    for (int iteration = 0;
         iteration < warmup_iterations;
         ++iteration) {
        launchMatrixAdd(
            pattern,
            block_count,
            threads_per_block,
            device_a,
            device_b,
            device_c,
            row_count,
            column_count
        );
    }

    checkCuda(
        cudaGetLastError(),
        "warmup launch"
    );

    checkCuda(
        cudaDeviceSynchronize(),
        "warmup execution"
    );

    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    checkCuda(
        cudaEventCreate(&start_event),
        "cudaEventCreate start_event"
    );

    checkCuda(
        cudaEventCreate(&stop_event),
        "cudaEventCreate stop_event"
    );

    checkCuda(
        cudaEventRecord(start_event),
        "cudaEventRecord start_event"
    );

    for (int iteration = 0;
         iteration < measured_iterations;
         ++iteration) {
        launchMatrixAdd(
            pattern,
            block_count,
            threads_per_block,
            device_a,
            device_b,
            device_c,
            row_count,
            column_count
        );
    }

    checkCuda(
        cudaGetLastError(),
        "measured launch"
    );

    checkCuda(
        cudaEventRecord(stop_event),
        "cudaEventRecord stop_event"
    );

    checkCuda(
        cudaEventSynchronize(stop_event),
        "cudaEventSynchronize stop_event"
    );

    float total_time_ms = 0.0F;

    checkCuda(
        cudaEventElapsedTime(
            &total_time_ms,
            start_event,
            stop_event
        ),
        "cudaEventElapsedTime"
    );

    checkCuda(
        cudaEventDestroy(start_event),
        "cudaEventDestroy start_event"
    );

    checkCuda(
        cudaEventDestroy(stop_event),
        "cudaEventDestroy stop_event"
    );

    return static_cast<double>(total_time_ms) /
           static_cast<double>(measured_iterations);
}

void verifyResult(
    const std::vector<int>& a,
    const std::vector<int>& b,
    const std::vector<int>& c,
    const std::string& label
) {
    for (std::size_t index = 0;
         index < c.size();
         ++index) {
        if (c[index] != a[index] + b[index]) {
            std::cerr
                << label
                << " wrong result at index "
                << index
                << '\n';

            std::exit(EXIT_FAILURE);
        }
    }
}

double calculateEffectiveBandwidth(
    const std::size_t byte_count,
    const double average_time_ms
) {
    const double bytes_per_iteration =
        static_cast<double>(byte_count) * 3.0;

    const double seconds_per_iteration =
        average_time_ms / 1000.0;

    return bytes_per_iteration /
           seconds_per_iteration /
           1.0e9;
}

int main() {
    // 方阵尺寸同时能被32和8整除，排除边界线程差异。
    const std::size_t row_count = 1024;
    const std::size_t column_count = 1024;
    const std::size_t element_count =
        row_count * column_count;
    const std::size_t byte_count =
        element_count * sizeof(int);

    const dim3 threads_per_block{32, 8};
    const dim3 block_count{
        static_cast<unsigned int>(
            column_count / threads_per_block.x
        ),
        static_cast<unsigned int>(
            row_count / threads_per_block.y
        )
    };

    const int warmup_iterations = 10;
    const int measured_iterations = 1000;

    std::vector<int> host_a(element_count);
    std::vector<int> host_b(element_count);
    std::vector<int> host_c(element_count, 0);

    for (std::size_t index = 0;
         index < element_count;
         ++index) {
        host_a[index] = static_cast<int>(index);
        host_b[index] = static_cast<int>(index * 2);
    }

    int* device_a = nullptr;
    int* device_b = nullptr;
    int* device_c = nullptr;

    checkCuda(
        cudaMalloc(&device_a, byte_count),
        "cudaMalloc device_a"
    );
    checkCuda(
        cudaMalloc(&device_b, byte_count),
        "cudaMalloc device_b"
    );
    checkCuda(
        cudaMalloc(&device_c, byte_count),
        "cudaMalloc device_c"
    );

    checkCuda(
        cudaMemcpy(
            device_a,
            host_a.data(),
            byte_count,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy host_a to device_a"
    );
    checkCuda(
        cudaMemcpy(
            device_b,
            host_b.data(),
            byte_count,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy host_b to device_b"
    );

    const double coalesced_time_ms =
        benchmarkMatrixAdd(
            AccessPattern::coalesced,
            block_count,
            threads_per_block,
            device_a,
            device_b,
            device_c,
            row_count,
            column_count,
            warmup_iterations,
            measured_iterations
        );

    checkCuda(
        cudaMemcpy(
            host_c.data(),
            device_c,
            byte_count,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy coalesced result"
    );
    verifyResult(host_a, host_b, host_c, "coalesced");

    const double strided_time_ms =
        benchmarkMatrixAdd(
            AccessPattern::strided,
            block_count,
            threads_per_block,
            device_a,
            device_b,
            device_c,
            row_count,
            column_count,
            warmup_iterations,
            measured_iterations
        );

    checkCuda(
        cudaMemcpy(
            host_c.data(),
            device_c,
            byte_count,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy strided result"
    );
    verifyResult(host_a, host_b, host_c, "strided");

    checkCuda(cudaFree(device_a), "cudaFree device_a");
    checkCuda(cudaFree(device_b), "cudaFree device_b");
    checkCuda(cudaFree(device_c), "cudaFree device_c");

    const double coalesced_bandwidth =
        calculateEffectiveBandwidth(
            byte_count,
            coalesced_time_ms
        );

    const double strided_bandwidth =
        calculateEffectiveBandwidth(
            byte_count,
            strided_time_ms
        );

    std::cout
        << "matrix shape: "
        << row_count
        << " rows x "
        << column_count
        << " columns\n"
        << "block dimensions: x="
        << threads_per_block.x
        << ", y="
        << threads_per_block.y
        << " ("
        << threads_per_block.x * threads_per_block.y
        << " threads)\n"
        << "coalesced average time: "
        << coalesced_time_ms
        << " ms\n"
        << "coalesced effective bandwidth: "
        << coalesced_bandwidth
        << " GB/s\n"
        << "strided average time: "
        << strided_time_ms
        << " ms\n"
        << "strided effective bandwidth: "
        << strided_bandwidth
        << " GB/s\n"
        << "strided/coalesced slowdown: "
        << strided_time_ms / coalesced_time_ms
        << "x\n"
        << "verification: both patterns passed\n";

    return 0;
}
