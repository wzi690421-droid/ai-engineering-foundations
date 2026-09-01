#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

constexpr unsigned int kBlockColumns = 16;
constexpr unsigned int kBlockRows = 16;

// 朴素GEMM：一个线程计算C中的一个元素。
__global__ void gemmNaive(
    const float* a,
    const float* b,
    float* c,
    const std::size_t row_count,
    const std::size_t inner_count,
    const std::size_t column_count
) {
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.y) *
        blockDim.y +
        threadIdx.y;

    const std::size_t column =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    // 最右侧和最下侧Block中的部分线程可能落在C之外。
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

std::vector<float> gemmSerial(
    const std::vector<float>& a,
    const std::vector<float>& b,
    const std::size_t row_count,
    const std::size_t inner_count,
    const std::size_t column_count
) {
    std::vector<float> c(row_count * column_count, 0.0F);

    for (std::size_t row = 0;
         row < row_count;
         ++row) {
        for (std::size_t column = 0;
             column < column_count;
             ++column) {
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
    }

    return c;
}

int main() {
    // 三个维度都故意不选16的倍数，用于验证Grid边界。
    const std::size_t row_count = 37;
    const std::size_t inner_count = 29;
    const std::size_t column_count = 41;

    const std::size_t a_element_count =
        row_count * inner_count;
    const std::size_t b_element_count =
        inner_count * column_count;
    const std::size_t c_element_count =
        row_count * column_count;

    std::vector<float> host_a(a_element_count);
    std::vector<float> host_b(b_element_count);
    std::vector<float> host_c(c_element_count, 0.0F);

    for (std::size_t index = 0;
         index < a_element_count;
         ++index) {
        host_a[index] =
            static_cast<float>(index % 7 + 1);
    }

    for (std::size_t index = 0;
         index < b_element_count;
         ++index) {
        host_b[index] =
            static_cast<float>(index % 5 + 1);
    }

    const std::vector<float> reference_c =
        gemmSerial(
            host_a,
            host_b,
            row_count,
            inner_count,
            column_count
        );

    const std::size_t a_byte_count =
        a_element_count * sizeof(float);
    const std::size_t b_byte_count =
        b_element_count * sizeof(float);
    const std::size_t c_byte_count =
        c_element_count * sizeof(float);

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;

    checkCuda(cudaMalloc(&device_a, a_byte_count), "cudaMalloc device_a");
    checkCuda(cudaMalloc(&device_b, b_byte_count), "cudaMalloc device_b");
    checkCuda(cudaMalloc(&device_c, c_byte_count), "cudaMalloc device_c");

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

    const dim3 threads_per_block{
        kBlockColumns,
        kBlockRows
    };
    const dim3 block_count{
        static_cast<unsigned int>(
            (column_count + threads_per_block.x - 1) /
            threads_per_block.x
        ),
        static_cast<unsigned int>(
            (row_count + threads_per_block.y - 1) /
            threads_per_block.y
        )
    };

    gemmNaive<<<block_count, threads_per_block>>>(
        device_a,
        device_b,
        device_c,
        row_count,
        inner_count,
        column_count
    );

    checkCuda(cudaGetLastError(), "gemmNaive launch");
    checkCuda(cudaDeviceSynchronize(), "gemmNaive execution");

    checkCuda(
        cudaMemcpy(
            host_c.data(),
            device_c,
            c_byte_count,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy device_c to host_c"
    );

    checkCuda(cudaFree(device_a), "cudaFree device_a");
    checkCuda(cudaFree(device_b), "cudaFree device_b");
    checkCuda(cudaFree(device_c), "cudaFree device_c");

    constexpr float tolerance = 1.0e-4F;
    float maximum_absolute_error = 0.0F;

    for (std::size_t index = 0;
         index < c_element_count;
         ++index) {
        const float absolute_error =
            std::fabs(host_c[index] - reference_c[index]);

        if (absolute_error > maximum_absolute_error) {
            maximum_absolute_error = absolute_error;
        }

        if (absolute_error > tolerance) {
            const std::size_t row = index / column_count;
            const std::size_t column = index % column_count;

            std::cerr
                << "verification failed at C["
                << row
                << "]["
                << column
                << "]: CPU="
                << reference_c[index]
                << ", GPU="
                << host_c[index]
                << '\n';

            return 1;
        }
    }

    std::cout
        << "A shape: "
        << row_count
        << " x "
        << inner_count
        << '\n'
        << "B shape: "
        << inner_count
        << " x "
        << column_count
        << '\n'
        << "C shape: "
        << row_count
        << " x "
        << column_count
        << '\n'
        << "block: ("
        << threads_per_block.x
        << ", "
        << threads_per_block.y
        << ")\n"
        << "grid: ("
        << block_count.x
        << ", "
        << block_count.y
        << ")\n"
        << "maximum absolute error: "
        << maximum_absolute_error
        << '\n'
        << "naive GEMM verification passed\n";

    return 0;
}
