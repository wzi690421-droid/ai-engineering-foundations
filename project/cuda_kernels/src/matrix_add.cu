#include <cuda_runtime.h>

#include <cstddef>
#include <vector>
#include <cstdlib>
#include <iostream>

__global__ void matrixAdd(
    const int* a,
    const int* b,
    int* c,
    const std::size_t row_count,
    const std::size_t column_count
) {
    // x方向负责列。
    const std::size_t column =
         blockIdx.x * blockDim.x + threadIdx.x;

    // y方向负责行。
    const std::size_t row =
         blockIdx.y * blockDim.y + threadIdx.y;

    // 最边缘的Block可能有部分线程落在矩阵外。
    if (row < row_count && column < column_count) {
        // 把二维坐标转换成连续内存中的一维下标。
        const std::size_t index =
            row * column_count + column;

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

void printElementCalculation(
    const std::vector<int>& a,
    const std::vector<int>& b,
    const std::vector<int>& c,
    const std::size_t row,
    const std::size_t column,
    const std::size_t column_count
) {
    const std::size_t index =
        row * column_count + column;

    std::cout
        << "  position ("
        << row
        << ", "
        << column
        << "), index "
        << index
        << ": "
        << a[index]
        << " + "
        << b[index]
        << " = "
        << c[index]
        << '\n';
}

int main() {
    // 故意选择不能被16整除的尺寸，
    // 用于验证矩阵边缘的线程不会越界。
    const std::size_t row_count = 1003;
    const std::size_t column_count = 997;

    const std::size_t element_count =
        row_count * column_count;

    const std::size_t byte_count =
        element_count * sizeof(int);

    // 矩阵底层仍然使用一维连续数组保存。
    std::vector<int> host_a(element_count);
    std::vector<int> host_b(element_count);
    std::vector<int> host_c(element_count, 0);

    // 初始化两个输入矩阵。
    for (std::size_t index = 0;
         index < element_count;
         ++index) {
        host_a[index] = static_cast<int>(index);
        host_b[index] = static_cast<int>(index * 2);
    }

    // 每个Block是16列、16行，共256个线程。
    const dim3 threads_per_block{
        32,
        8
    };

    // Grid的x方向覆盖所有列，y方向覆盖所有行。
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

// 保存三个GPU数组的地址。
int* device_a = nullptr;
int* device_b = nullptr;
int* device_c = nullptr;

// 为三个矩阵申请GPU显存。
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

// 把两个输入矩阵复制到GPU。
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

const int warmup_iterations = 10;
const int measured_iterations = 1000;

// 先预热Kernel，避免首次启动的一次性开销污染正式结果。
for (int iteration = 0;
     iteration < warmup_iterations;
     ++iteration) {
    matrixAdd<<<block_count, threads_per_block>>>(
        device_a,
        device_b,
        device_c,
        row_count,
        column_count
    );
}

checkCuda(
    cudaGetLastError(),
    "matrixAdd warmup launch"
);

checkCuda(
    cudaDeviceSynchronize(),
    "matrixAdd warmup execution"
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

// Event只包围Kernel循环，不包含H2D、D2H和CPU验证。
checkCuda(
    cudaEventRecord(start_event),
    "cudaEventRecord start_event"
);

for (int iteration = 0;
     iteration < measured_iterations;
     ++iteration) {
    matrixAdd<<<block_count, threads_per_block>>>(
        device_a,
        device_b,
        device_c,
        row_count,
        column_count
    );
}

checkCuda(
    cudaGetLastError(),
    "matrixAdd measured launch"
);

checkCuda(
    cudaEventRecord(stop_event),
    "cudaEventRecord stop_event"
);

checkCuda(
    cudaEventSynchronize(stop_event),
    "cudaEventSynchronize stop_event"
);

float total_kernel_time_ms = 0.0F;

checkCuda(
    cudaEventElapsedTime(
        &total_kernel_time_ms,
        start_event,
        stop_event
    ),
    "cudaEventElapsedTime matrixAdd"
);

checkCuda(
    cudaEventDestroy(start_event),
    "cudaEventDestroy start_event"
);

checkCuda(
    cudaEventDestroy(stop_event),
    "cudaEventDestroy stop_event"
);

const double average_kernel_time_ms =
    static_cast<double>(total_kernel_time_ms) /
    static_cast<double>(measured_iterations);

// 每个元素读取a、读取b、写入c，共移动3个int。
const double measured_byte_count =
    static_cast<double>(byte_count) *
    3.0 *
    static_cast<double>(measured_iterations);

const double measured_time_seconds =
    static_cast<double>(total_kernel_time_ms) /
    1000.0;

const double effective_bandwidth_gb_per_second =
    measured_byte_count /
    measured_time_seconds /
    1.0e9;

// 把计算结果复制回CPU。
checkCuda(
    cudaMemcpy(
        host_c.data(),
        device_c,
        byte_count,
        cudaMemcpyDeviceToHost
    ),
    "cudaMemcpy device_c to host_c"
);

// 释放GPU显存。
checkCuda(
    cudaFree(device_a),
    "cudaFree device_a"
);

checkCuda(
    cudaFree(device_b),
    "cudaFree device_b"
);

checkCuda(
    cudaFree(device_c),
    "cudaFree device_c"
);

// 使用CPU计算结果，检查所有矩阵元素。
for (std::size_t index = 0;
     index < element_count;
     ++index) {
    const int expected =
        host_a[index] + host_b[index];

    if (host_c[index] != expected) {
        std::cerr
            << "wrong result at index "
            << index
            << '\n';

        return 1;
    }
}

const std::size_t launched_thread_count =
    static_cast<std::size_t>(block_count.x) *
    static_cast<std::size_t>(block_count.y) *
    static_cast<std::size_t>(threads_per_block.x) *
    static_cast<std::size_t>(threads_per_block.y);

std::cout
    << "matrix shape: "
    << row_count
    << " x "
    << column_count
    << '\n'
    << "block shape: "
    << threads_per_block.x
    << " x "
    << threads_per_block.y
    << '\n'
    << "grid shape: "
    << block_count.x
    << " x "
    << block_count.y
    << '\n'
    << "matrix elements: "
    << element_count
    << '\n'
    << "launched threads: "
    << launched_thread_count
    << '\n'
    << "boundary threads skipped: "
    << launched_thread_count - element_count
    << '\n'
    << "warmup iterations: "
    << warmup_iterations
    << '\n'
    << "measured iterations: "
    << measured_iterations
    << '\n'
    << "average kernel time: "
    << average_kernel_time_ms
    << " ms\n"
    << "effective bandwidth: "
    << effective_bandwidth_gb_per_second
    << " GB/s\n"
    << "sample calculations:\n";

printElementCalculation(
    host_a,
    host_b,
    host_c,
    0,
    0,
    column_count
);

printElementCalculation(
    host_a,
    host_b,
    host_c,
    row_count / 2,
    column_count / 2,
    column_count
);

printElementCalculation(
    host_a,
    host_b,
    host_c,
    row_count - 1,
    column_count - 1,
    column_count
);

std::cout
    << "verification: all "
    << element_count
    << " elements passed\n";

    return 0;
}
