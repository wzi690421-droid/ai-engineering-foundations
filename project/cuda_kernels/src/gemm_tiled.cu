#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

constexpr unsigned int kTileSize = 16;

__global__ void gemmNaiveCoalesced(
    const float* a,
    const float* b,
    float* c,
    const std::size_t row_count,
    const std::size_t inner_count,
    const std::size_t column_count
) {
    const std::size_t column =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.y) *
        blockDim.y +
        threadIdx.y;

    if (row >= row_count || column >= column_count) {
        return;
    }

    float sum = 0.0F;

    for (std::size_t inner = 0;
         inner < inner_count;
         ++inner) {
        sum +=
            a[row * inner_count + inner] *
            b[inner * column_count + column];
    }

    c[row * column_count + column] = sum;
}

__global__ void gemmTiled(
    const float* a,
    const float* b,
    float* c,
    const std::size_t row_count,
    const std::size_t inner_count,
    const std::size_t column_count
) {
    __shared__ float shared_a[kTileSize][kTileSize];
    __shared__ float shared_b[kTileSize][kTileSize];

    const unsigned int local_column = threadIdx.x;
    const unsigned int local_row = threadIdx.y;
    const std::size_t column =
        static_cast<std::size_t>(blockIdx.x) *
        kTileSize +
        local_column;
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.y) *
        kTileSize +
        local_row;

    float sum = 0.0F;

    for (std::size_t tile_start = 0;
         tile_start < inner_count;
         tile_start += kTileSize) {
        const std::size_t a_column =
            tile_start + local_column;
        const std::size_t b_row =
            tile_start + local_row;

        // 所有线程都写入自己的Shared Memory位置；越界位置补零。
        shared_a[local_row][local_column] =
            row < row_count && a_column < inner_count
                ? a[row * inner_count + a_column]
                : 0.0F;

        shared_b[local_row][local_column] =
            b_row < inner_count && column < column_count
                ? b[b_row * column_count + column]
                : 0.0F;

        // 等待当前A、B Tile全部加载完成。
        __syncthreads();

        #pragma unroll
        for (unsigned int inner = 0;
             inner < kTileSize;
             ++inner) {
            sum +=
                shared_a[local_row][inner] *
                shared_b[inner][local_column];
        }

        // 等待所有线程使用完当前Tile，再覆盖Shared Memory。
        __syncthreads();
    }

    if (row < row_count && column < column_count) {
        c[row * column_count + column] = sum;
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

void launchGemm(
    const bool use_tiled,
    const dim3& grid,
    const dim3& block,
    const float* a,
    const float* b,
    float* c,
    const std::size_t row_count,
    const std::size_t inner_count,
    const std::size_t column_count
) {
    if (use_tiled) {
        gemmTiled<<<grid, block>>>(
            a,
            b,
            c,
            row_count,
            inner_count,
            column_count
        );
    } else {
        gemmNaiveCoalesced<<<grid, block>>>(
            a,
            b,
            c,
            row_count,
            inner_count,
            column_count
        );
    }
}

float measureGemmMilliseconds(
    const bool use_tiled,
    const unsigned int iterations,
    const dim3& grid,
    const dim3& block,
    const float* a,
    const float* b,
    float* c,
    const std::size_t row_count,
    const std::size_t inner_count,
    const std::size_t column_count
) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    checkCuda(cudaEventRecord(start), "cudaEventRecord start");

    for (unsigned int iteration = 0;
         iteration < iterations;
         ++iteration) {
        launchGemm(
            use_tiled,
            grid,
            block,
            a,
            b,
            c,
            row_count,
            inner_count,
            column_count
        );
    }

    checkCuda(
        cudaGetLastError(),
        use_tiled
            ? "gemmTiled measured launch"
            : "gemmNaiveCoalesced measured launch"
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
    // 三个维度均不是Tile大小的倍数，用于验证补零和边界写入。
    const std::size_t row_count = 513;
    const std::size_t inner_count = 509;
    const std::size_t column_count = 517;

    const std::size_t a_element_count =
        row_count * inner_count;
    const std::size_t b_element_count =
        inner_count * column_count;
    const std::size_t c_element_count =
        row_count * column_count;

    std::vector<float> host_a(a_element_count);
    std::vector<float> host_b(b_element_count);
    std::vector<float> host_naive_c(c_element_count);
    std::vector<float> host_tiled_c(c_element_count);

    for (std::size_t index = 0;
         index < a_element_count;
         ++index) {
        host_a[index] =
            static_cast<float>(index % 7 + 1) *
            0.125F;
    }

    for (std::size_t index = 0;
         index < b_element_count;
         ++index) {
        host_b[index] =
            static_cast<float>(index % 5 + 1) *
            0.25F;
    }

    const std::size_t a_byte_count =
        a_element_count * sizeof(float);
    const std::size_t b_byte_count =
        b_element_count * sizeof(float);
    const std::size_t c_byte_count =
        c_element_count * sizeof(float);

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_naive_c = nullptr;
    float* device_tiled_c = nullptr;

    checkCuda(cudaMalloc(&device_a, a_byte_count), "cudaMalloc device_a");
    checkCuda(cudaMalloc(&device_b, b_byte_count), "cudaMalloc device_b");
    checkCuda(
        cudaMalloc(&device_naive_c, c_byte_count),
        "cudaMalloc device_naive_c"
    );
    checkCuda(
        cudaMalloc(&device_tiled_c, c_byte_count),
        "cudaMalloc device_tiled_c"
    );

    checkCuda(
        cudaMemcpy(
            device_a,
            host_a.data(),
            a_byte_count,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy host_a to device_a"
    );
    checkCuda(
        cudaMemcpy(
            device_b,
            host_b.data(),
            b_byte_count,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy host_b to device_b"
    );

    const dim3 block{kTileSize, kTileSize};
    const dim3 grid{
        static_cast<unsigned int>(
            (column_count + block.x - 1) /
            block.x
        ),
        static_cast<unsigned int>(
            (row_count + block.y - 1) /
            block.y
        )
    };

    // 先分别运行一次并比较完整输出。
    launchGemm(
        false,
        grid,
        block,
        device_a,
        device_b,
        device_naive_c,
        row_count,
        inner_count,
        column_count
    );
    launchGemm(
        true,
        grid,
        block,
        device_a,
        device_b,
        device_tiled_c,
        row_count,
        inner_count,
        column_count
    );

    checkCuda(cudaGetLastError(), "GEMM verification launch");
    checkCuda(cudaDeviceSynchronize(), "GEMM verification execution");

    checkCuda(
        cudaMemcpy(
            host_naive_c.data(),
            device_naive_c,
            c_byte_count,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy naive C to host"
    );
    checkCuda(
        cudaMemcpy(
            host_tiled_c.data(),
            device_tiled_c,
            c_byte_count,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy tiled C to host"
    );

    constexpr float tolerance = 1.0e-3F;
    float maximum_absolute_error = 0.0F;

    for (std::size_t index = 0;
         index < c_element_count;
         ++index) {
        const float absolute_error = std::fabs(
            host_naive_c[index] -
            host_tiled_c[index]
        );

        if (absolute_error > maximum_absolute_error) {
            maximum_absolute_error = absolute_error;
        }

        if (absolute_error > tolerance) {
            std::cerr
                << "verification failed at index "
                << index
                << ": naive="
                << host_naive_c[index]
                << ", tiled="
                << host_tiled_c[index]
                << '\n';

            return 1;
        }
    }

    constexpr unsigned int warmup_iterations = 20;
    constexpr unsigned int measured_iterations = 100;
    constexpr unsigned int measured_rounds = 5;

    for (unsigned int iteration = 0;
         iteration < warmup_iterations;
         ++iteration) {
        launchGemm(
            false,
            grid,
            block,
            device_a,
            device_b,
            device_naive_c,
            row_count,
            inner_count,
            column_count
        );
        launchGemm(
            true,
            grid,
            block,
            device_a,
            device_b,
            device_tiled_c,
            row_count,
            inner_count,
            column_count
        );
    }

    checkCuda(cudaGetLastError(), "GEMM warmup launch");
    checkCuda(cudaDeviceSynchronize(), "GEMM warmup execution");

    double naive_total_milliseconds = 0.0;
    double tiled_total_milliseconds = 0.0;

    for (unsigned int round = 0;
         round < measured_rounds;
         ++round) {
        if (round % 2 == 0) {
            naive_total_milliseconds += measureGemmMilliseconds(
                false,
                measured_iterations,
                grid,
                block,
                device_a,
                device_b,
                device_naive_c,
                row_count,
                inner_count,
                column_count
            );
            tiled_total_milliseconds += measureGemmMilliseconds(
                true,
                measured_iterations,
                grid,
                block,
                device_a,
                device_b,
                device_tiled_c,
                row_count,
                inner_count,
                column_count
            );
        } else {
            tiled_total_milliseconds += measureGemmMilliseconds(
                true,
                measured_iterations,
                grid,
                block,
                device_a,
                device_b,
                device_tiled_c,
                row_count,
                inner_count,
                column_count
            );
            naive_total_milliseconds += measureGemmMilliseconds(
                false,
                measured_iterations,
                grid,
                block,
                device_a,
                device_b,
                device_naive_c,
                row_count,
                inner_count,
                column_count
            );
        }
    }

    const double naive_average_milliseconds =
        naive_total_milliseconds /
        static_cast<double>(measured_rounds);
    const double tiled_average_milliseconds =
        tiled_total_milliseconds /
        static_cast<double>(measured_rounds);
    const double operation_count =
        2.0 *
        static_cast<double>(row_count) *
        static_cast<double>(inner_count) *
        static_cast<double>(column_count);

    const double naive_gflops =
        operation_count /
        (naive_average_milliseconds * 1.0e6);
    const double tiled_gflops =
        operation_count /
        (tiled_average_milliseconds * 1.0e6);

    checkCuda(cudaFree(device_a), "cudaFree device_a");
    checkCuda(cudaFree(device_b), "cudaFree device_b");
    checkCuda(cudaFree(device_naive_c), "cudaFree device_naive_c");
    checkCuda(cudaFree(device_tiled_c), "cudaFree device_tiled_c");

    std::cout
        << "matrix shape: "
        << row_count
        << " x "
        << inner_count
        << " times "
        << inner_count
        << " x "
        << column_count
        << '\n'
        << "maximum absolute error: "
        << maximum_absolute_error
        << '\n'
        << std::fixed
        << std::setprecision(3)
        << "naive coalesced average: "
        << naive_average_milliseconds
        << " ms, "
        << naive_gflops
        << " GFLOP/s\n"
        << "tiled average: "
        << tiled_average_milliseconds
        << " ms, "
        << tiled_gflops
        << " GFLOP/s\n"
        << "tiled speedup: "
        << naive_average_milliseconds /
            tiled_average_milliseconds
        << "x\n"
        << "tiled GEMM verification passed\n";

    return 0;
}
