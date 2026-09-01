#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

constexpr unsigned int kTileSize = 16;

__global__ void gemmTiledForCublasComparison(
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

        shared_a[local_row][local_column] =
            row < row_count && a_column < inner_count
                ? a[row * inner_count + a_column]
                : 0.0F;

        shared_b[local_row][local_column] =
            b_row < inner_count && column < column_count
                ? b[b_row * column_count + column]
                : 0.0F;

        __syncthreads();

        #pragma unroll
        for (unsigned int inner = 0;
             inner < kTileSize;
             ++inner) {
            sum +=
                shared_a[local_row][inner] *
                shared_b[inner][local_column];
        }

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

void checkCublas(
    const cublasStatus_t status,
    const char* operation
) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr
            << operation
            << " failed: "
            << cublasGetStatusString(status)
            << '\n';

        std::exit(EXIT_FAILURE);
    }
}

void launchTiled(
    const dim3& grid,
    const dim3& block,
    const float* a,
    const float* b,
    float* c,
    const std::size_t row_count,
    const std::size_t inner_count,
    const std::size_t column_count
) {
    gemmTiledForCublasComparison<<<grid, block>>>(
        a,
        b,
        c,
        row_count,
        inner_count,
        column_count
    );
}

void launchCublasRowMajor(
    cublasHandle_t handle,
    const float* a,
    const float* b,
    float* c,
    const std::size_t row_count,
    const std::size_t inner_count,
    const std::size_t column_count
) {
    const float alpha = 1.0F;
    const float beta = 0.0F;

    // cuBLAS按列主序解释数据。行主序C=A*B在相同内存中等价于：
    // C^T = B^T * A^T，因此交换A、B并交换输出的行列数。
    checkCublas(
        cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            static_cast<int>(column_count),
            static_cast<int>(row_count),
            static_cast<int>(inner_count),
            &alpha,
            b,
            CUDA_R_32F,
            static_cast<int>(column_count),
            a,
            CUDA_R_32F,
            static_cast<int>(inner_count),
            &beta,
            c,
            CUDA_R_32F,
            static_cast<int>(column_count),
            CUBLAS_COMPUTE_32F_PEDANTIC,
            CUBLAS_GEMM_DEFAULT
        ),
        "cublasGemmEx FP32 pedantic"
    );
}

float measureTiledMilliseconds(
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

    checkCuda(cudaEventCreate(&start), "cudaEventCreate tiled start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate tiled stop");
    checkCuda(cudaEventRecord(start), "cudaEventRecord tiled start");

    for (unsigned int iteration = 0;
         iteration < iterations;
         ++iteration) {
        launchTiled(
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

    checkCuda(cudaGetLastError(), "tiled measured launch");
    checkCuda(cudaEventRecord(stop), "cudaEventRecord tiled stop");
    checkCuda(
        cudaEventSynchronize(stop),
        "cudaEventSynchronize tiled stop"
    );

    float total_milliseconds = 0.0F;
    checkCuda(
        cudaEventElapsedTime(
            &total_milliseconds,
            start,
            stop
        ),
        "cudaEventElapsedTime tiled"
    );

    checkCuda(cudaEventDestroy(start), "cudaEventDestroy tiled start");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy tiled stop");

    return total_milliseconds /
        static_cast<float>(iterations);
}

float measureCublasMilliseconds(
    const unsigned int iterations,
    cublasHandle_t handle,
    const float* a,
    const float* b,
    float* c,
    const std::size_t row_count,
    const std::size_t inner_count,
    const std::size_t column_count
) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    checkCuda(cudaEventCreate(&start), "cudaEventCreate cuBLAS start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate cuBLAS stop");
    checkCuda(cudaEventRecord(start), "cudaEventRecord cuBLAS start");

    for (unsigned int iteration = 0;
         iteration < iterations;
         ++iteration) {
        launchCublasRowMajor(
            handle,
            a,
            b,
            c,
            row_count,
            inner_count,
            column_count
        );
    }

    checkCuda(cudaEventRecord(stop), "cudaEventRecord cuBLAS stop");
    checkCuda(
        cudaEventSynchronize(stop),
        "cudaEventSynchronize cuBLAS stop"
    );

    float total_milliseconds = 0.0F;
    checkCuda(
        cudaEventElapsedTime(
            &total_milliseconds,
            start,
            stop
        ),
        "cudaEventElapsedTime cuBLAS"
    );

    checkCuda(cudaEventDestroy(start), "cudaEventDestroy cuBLAS start");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy cuBLAS stop");

    return total_milliseconds /
        static_cast<float>(iterations);
}

int main() {
    const std::size_t row_count = 1'024;
    const std::size_t inner_count = 1'024;
    const std::size_t column_count = 1'024;

    const std::size_t a_element_count =
        row_count * inner_count;
    const std::size_t b_element_count =
        inner_count * column_count;
    const std::size_t c_element_count =
        row_count * column_count;

    std::vector<float> host_a(a_element_count);
    std::vector<float> host_b(b_element_count);
    std::vector<float> host_tiled_c(c_element_count);
    std::vector<float> host_cublas_c(c_element_count);

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
    float* device_tiled_c = nullptr;
    float* device_cublas_c = nullptr;

    checkCuda(cudaMalloc(&device_a, a_byte_count), "cudaMalloc device_a");
    checkCuda(cudaMalloc(&device_b, b_byte_count), "cudaMalloc device_b");
    checkCuda(
        cudaMalloc(&device_tiled_c, c_byte_count),
        "cudaMalloc device_tiled_c"
    );
    checkCuda(
        cudaMalloc(&device_cublas_c, c_byte_count),
        "cudaMalloc device_cublas_c"
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

    cublasHandle_t handle = nullptr;
    checkCublas(cublasCreate(&handle), "cublasCreate");
    checkCublas(
        cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH),
        "cublasSetMathMode pedantic"
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

    // 正确性验证不计时。
    launchTiled(
        grid,
        block,
        device_a,
        device_b,
        device_tiled_c,
        row_count,
        inner_count,
        column_count
    );
    checkCuda(cudaGetLastError(), "tiled verification launch");
    launchCublasRowMajor(
        handle,
        device_a,
        device_b,
        device_cublas_c,
        row_count,
        inner_count,
        column_count
    );
    checkCuda(cudaDeviceSynchronize(), "GEMM verification execution");

    checkCuda(
        cudaMemcpy(
            host_tiled_c.data(),
            device_tiled_c,
            c_byte_count,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy tiled C to host"
    );
    checkCuda(
        cudaMemcpy(
            host_cublas_c.data(),
            device_cublas_c,
            c_byte_count,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy cuBLAS C to host"
    );

    constexpr float absolute_tolerance = 1.0e-3F;
    constexpr float relative_tolerance = 1.0e-4F;
    float maximum_absolute_error = 0.0F;

    for (std::size_t index = 0;
         index < c_element_count;
         ++index) {
        const float absolute_error = std::fabs(
            host_tiled_c[index] -
            host_cublas_c[index]
        );
        const float allowed_error =
            absolute_tolerance +
            relative_tolerance *
                std::fabs(host_cublas_c[index]);

        if (absolute_error > maximum_absolute_error) {
            maximum_absolute_error = absolute_error;
        }

        if (absolute_error > allowed_error) {
            std::cerr
                << "verification failed at index "
                << index
                << ": tiled="
                << host_tiled_c[index]
                << ", cuBLAS="
                << host_cublas_c[index]
                << ", allowed error="
                << allowed_error
                << '\n';

            return 1;
        }
    }

    constexpr unsigned int warmup_iterations = 20;
    constexpr unsigned int measured_iterations = 50;
    constexpr unsigned int measured_rounds = 5;

    for (unsigned int iteration = 0;
         iteration < warmup_iterations;
         ++iteration) {
        launchTiled(
            grid,
            block,
            device_a,
            device_b,
            device_tiled_c,
            row_count,
            inner_count,
            column_count
        );
        launchCublasRowMajor(
            handle,
            device_a,
            device_b,
            device_cublas_c,
            row_count,
            inner_count,
            column_count
        );
    }

    checkCuda(cudaGetLastError(), "GEMM warmup launch");
    checkCuda(cudaDeviceSynchronize(), "GEMM warmup execution");

    double tiled_total_milliseconds = 0.0;
    double cublas_total_milliseconds = 0.0;

    for (unsigned int round = 0;
         round < measured_rounds;
         ++round) {
        if (round % 2 == 0) {
            tiled_total_milliseconds += measureTiledMilliseconds(
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
            cublas_total_milliseconds += measureCublasMilliseconds(
                measured_iterations,
                handle,
                device_a,
                device_b,
                device_cublas_c,
                row_count,
                inner_count,
                column_count
            );
        } else {
            cublas_total_milliseconds += measureCublasMilliseconds(
                measured_iterations,
                handle,
                device_a,
                device_b,
                device_cublas_c,
                row_count,
                inner_count,
                column_count
            );
            tiled_total_milliseconds += measureTiledMilliseconds(
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
        }
    }

    const double tiled_average_milliseconds =
        tiled_total_milliseconds /
        static_cast<double>(measured_rounds);
    const double cublas_average_milliseconds =
        cublas_total_milliseconds /
        static_cast<double>(measured_rounds);
    const double operation_count =
        2.0 *
        static_cast<double>(row_count) *
        static_cast<double>(inner_count) *
        static_cast<double>(column_count);

    const double tiled_gflops =
        operation_count /
        (tiled_average_milliseconds * 1.0e6);
    const double cublas_gflops =
        operation_count /
        (cublas_average_milliseconds * 1.0e6);

    checkCublas(cublasDestroy(handle), "cublasDestroy");
    checkCuda(cudaFree(device_a), "cudaFree device_a");
    checkCuda(cudaFree(device_b), "cudaFree device_b");
    checkCuda(cudaFree(device_tiled_c), "cudaFree device_tiled_c");
    checkCuda(cudaFree(device_cublas_c), "cudaFree device_cublas_c");

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
        << "cuBLAS compute mode: FP32 pedantic\n"
        << "maximum absolute error: "
        << maximum_absolute_error
        << '\n'
        << std::fixed
        << std::setprecision(3)
        << "tiled average: "
        << tiled_average_milliseconds
        << " ms, "
        << tiled_gflops
        << " GFLOP/s\n"
        << "cuBLAS average: "
        << cublas_average_milliseconds
        << " ms, "
        << cublas_gflops
        << " GFLOP/s\n"
        << "cuBLAS speedup over tiled: "
        << tiled_average_milliseconds /
            cublas_average_milliseconds
        << "x\n"
        << "cuBLAS comparison verification passed\n";

    return 0;
}
