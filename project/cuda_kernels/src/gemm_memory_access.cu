#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

constexpr unsigned int kBlockX = 32;
constexpr unsigned int kBlockY = 8;

// 对照版：Warp内连续的threadIdx.x对应不同的行。
// A的读取和C的写入都以整行宽度为步长。
__global__ void gemmStridedMapping(
    const float* a,
    const float* b,
    float* c,
    const std::size_t row_count,
    const std::size_t inner_count,
    const std::size_t column_count
) {
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;
    const std::size_t column =
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

// 合并访存版：Warp内连续的threadIdx.x对应同一行的连续列。
// B的读取和C的写入在Warp内连续，A的读取为相同地址广播。
__global__ void gemmCoalescedMapping(
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
    const bool use_coalesced_mapping,
    const dim3& strided_grid,
    const dim3& coalesced_grid,
    const dim3& threads_per_block,
    const float* a,
    const float* b,
    float* c,
    const std::size_t row_count,
    const std::size_t inner_count,
    const std::size_t column_count
) {
    if (use_coalesced_mapping) {
        gemmCoalescedMapping<<<
            coalesced_grid,
            threads_per_block
        >>>(
            a,
            b,
            c,
            row_count,
            inner_count,
            column_count
        );
    } else {
        gemmStridedMapping<<<
            strided_grid,
            threads_per_block
        >>>(
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
    const bool use_coalesced_mapping,
    const unsigned int iterations,
    const dim3& strided_grid,
    const dim3& coalesced_grid,
    const dim3& threads_per_block,
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
            use_coalesced_mapping,
            strided_grid,
            coalesced_grid,
            threads_per_block,
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
        use_coalesced_mapping
            ? "gemmCoalescedMapping measured launch"
            : "gemmStridedMapping measured launch"
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
    const std::size_t row_count = 512;
    const std::size_t inner_count = 512;
    const std::size_t column_count = 512;

    const std::size_t a_element_count =
        row_count * inner_count;
    const std::size_t b_element_count =
        inner_count * column_count;
    const std::size_t c_element_count =
        row_count * column_count;

    std::vector<float> host_a(a_element_count);
    std::vector<float> host_b(b_element_count);
    std::vector<float> host_strided_c(c_element_count);
    std::vector<float> host_coalesced_c(c_element_count);

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
    float* device_strided_c = nullptr;
    float* device_coalesced_c = nullptr;

    checkCuda(cudaMalloc(&device_a, a_byte_count), "cudaMalloc device_a");
    checkCuda(cudaMalloc(&device_b, b_byte_count), "cudaMalloc device_b");
    checkCuda(
        cudaMalloc(&device_strided_c, c_byte_count),
        "cudaMalloc device_strided_c"
    );
    checkCuda(
        cudaMalloc(&device_coalesced_c, c_byte_count),
        "cudaMalloc device_coalesced_c"
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

    const dim3 threads_per_block{kBlockX, kBlockY};
    const dim3 strided_grid{
        static_cast<unsigned int>(
            (row_count + threads_per_block.x - 1) /
            threads_per_block.x
        ),
        static_cast<unsigned int>(
            (column_count + threads_per_block.y - 1) /
            threads_per_block.y
        )
    };
    const dim3 coalesced_grid{
        static_cast<unsigned int>(
            (column_count + threads_per_block.x - 1) /
            threads_per_block.x
        ),
        static_cast<unsigned int>(
            (row_count + threads_per_block.y - 1) /
            threads_per_block.y
        )
    };

    // 先各执行一次并比较完整输出，保证线程映射只改变性能。
    launchGemm(
        false,
        strided_grid,
        coalesced_grid,
        threads_per_block,
        device_a,
        device_b,
        device_strided_c,
        row_count,
        inner_count,
        column_count
    );
    launchGemm(
        true,
        strided_grid,
        coalesced_grid,
        threads_per_block,
        device_a,
        device_b,
        device_coalesced_c,
        row_count,
        inner_count,
        column_count
    );

    checkCuda(cudaGetLastError(), "GEMM verification launch");
    checkCuda(cudaDeviceSynchronize(), "GEMM verification execution");

    checkCuda(
        cudaMemcpy(
            host_strided_c.data(),
            device_strided_c,
            c_byte_count,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy strided C to host"
    );
    checkCuda(
        cudaMemcpy(
            host_coalesced_c.data(),
            device_coalesced_c,
            c_byte_count,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy coalesced C to host"
    );

    constexpr float tolerance = 1.0e-4F;
    float maximum_absolute_error = 0.0F;

    for (std::size_t index = 0;
         index < c_element_count;
         ++index) {
        const float absolute_error = std::fabs(
            host_strided_c[index] -
            host_coalesced_c[index]
        );

        if (absolute_error > maximum_absolute_error) {
            maximum_absolute_error = absolute_error;
        }

        if (absolute_error > tolerance) {
            std::cerr
                << "verification failed at index "
                << index
                << ": strided="
                << host_strided_c[index]
                << ", coalesced="
                << host_coalesced_c[index]
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
            strided_grid,
            coalesced_grid,
            threads_per_block,
            device_a,
            device_b,
            device_strided_c,
            row_count,
            inner_count,
            column_count
        );
        launchGemm(
            true,
            strided_grid,
            coalesced_grid,
            threads_per_block,
            device_a,
            device_b,
            device_coalesced_c,
            row_count,
            inner_count,
            column_count
        );
    }

    checkCuda(cudaGetLastError(), "GEMM warmup launch");
    checkCuda(cudaDeviceSynchronize(), "GEMM warmup execution");

    double strided_total_milliseconds = 0.0;
    double coalesced_total_milliseconds = 0.0;

    for (unsigned int round = 0;
         round < measured_rounds;
         ++round) {
        if (round % 2 == 0) {
            strided_total_milliseconds += measureGemmMilliseconds(
                false,
                measured_iterations,
                strided_grid,
                coalesced_grid,
                threads_per_block,
                device_a,
                device_b,
                device_strided_c,
                row_count,
                inner_count,
                column_count
            );
            coalesced_total_milliseconds += measureGemmMilliseconds(
                true,
                measured_iterations,
                strided_grid,
                coalesced_grid,
                threads_per_block,
                device_a,
                device_b,
                device_coalesced_c,
                row_count,
                inner_count,
                column_count
            );
        } else {
            coalesced_total_milliseconds += measureGemmMilliseconds(
                true,
                measured_iterations,
                strided_grid,
                coalesced_grid,
                threads_per_block,
                device_a,
                device_b,
                device_coalesced_c,
                row_count,
                inner_count,
                column_count
            );
            strided_total_milliseconds += measureGemmMilliseconds(
                false,
                measured_iterations,
                strided_grid,
                coalesced_grid,
                threads_per_block,
                device_a,
                device_b,
                device_strided_c,
                row_count,
                inner_count,
                column_count
            );
        }
    }

    const double strided_average_milliseconds =
        strided_total_milliseconds /
        static_cast<double>(measured_rounds);
    const double coalesced_average_milliseconds =
        coalesced_total_milliseconds /
        static_cast<double>(measured_rounds);

    checkCuda(cudaFree(device_a), "cudaFree device_a");
    checkCuda(cudaFree(device_b), "cudaFree device_b");
    checkCuda(cudaFree(device_strided_c), "cudaFree device_strided_c");
    checkCuda(cudaFree(device_coalesced_c), "cudaFree device_coalesced_c");

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
        << "strided mapping average: "
        << strided_average_milliseconds
        << " ms\n"
        << "coalesced mapping average: "
        << coalesced_average_milliseconds
        << " ms\n"
        << "coalesced mapping speedup: "
        << strided_average_milliseconds /
            coalesced_average_milliseconds
        << "x\n"
        << "GEMM mapping verification passed\n";

    return 0;
}
