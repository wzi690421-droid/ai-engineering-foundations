#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <vector>

// Vector Add：每个线程负责计算一个位置。
__global__ void vectorAdd(
    const int* a,
    const int* b,
    int* c,
    const std::size_t count
) {
    // 当前线程负责的数组下标。
    std::size_t index =
        blockIdx.x * blockDim.x + threadIdx.x;

    // 整个Grid中一共有多少个线程。
    const std::size_t stride =
        blockDim.x * gridDim.x;

    // 一个线程每次跨过整个Grid的线程总数，从而继续处理后面的元素。
    for (; index < count; index += stride) {
        c[index] = a[index] + b[index];
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
    // 创建10'000'000个元素的输入和输出数组。
    const std::size_t element_count = 10'000'000;

    std::vector<int> host_a(element_count);
    std::vector<int> host_b(element_count);
    std::vector<int> host_c(element_count, 0);

    // 生成容易验证的输入数据。
    for (std::size_t index = 0; index < element_count; ++index) {
        host_a[index] = static_cast<int>(index);
        host_b[index] = static_cast<int>(index * 2);
    }

    // 三个数组长度相同，因此共用一套元素数量和字节数。
    const std::size_t count = host_a.size();
    const std::size_t byte_count = count * sizeof(int);

    // 把三个std::vector占用的主机内存注册为Pinned Memory。
    checkCuda(
        cudaHostRegister(
            host_a.data(),
            byte_count,
            cudaHostRegisterDefault
        ),
        "cudaHostRegister host_a"
    );

    checkCuda(
        cudaHostRegister(
            host_b.data(),
            byte_count,
            cudaHostRegisterDefault
        ),
        "cudaHostRegister host_b"
    );

    checkCuda(
        cudaHostRegister(
            host_c.data(),
            byte_count,
            cudaHostRegisterDefault
        ),
        "cudaHostRegister host_c"
    );

    // 用于保存GPU显存地址。
    int* device_a = nullptr;
    int* device_b = nullptr;
    int* device_c = nullptr;

    // 在GPU上申请显存。
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

    // 所有GPU计时实验复用同一对CUDA Event。
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    checkCuda(
        cudaEventCreate(&start_event),
        "cudaEventCreate start"
    );

    checkCuda(
        cudaEventCreate(&stop_event),
        "cudaEventCreate stop"
    );

    // 四组实验统一使用相同的预热次数和正式测量次数。
    const int warmup_iterations = 10;
    const int measured_iterations = 1000;

    // 预热CPU到GPU的两次输入复制，不计入正式结果。
    for (int iteration = 0;
         iteration < warmup_iterations;
         ++iteration) {
        checkCuda(
            cudaMemcpy(
                device_a,
                host_a.data(),
                byte_count,
                cudaMemcpyHostToDevice
            ),
            "H2D warmup host_a"
        );

        checkCuda(
            cudaMemcpy(
                device_b,
                host_b.data(),
                byte_count,
                cudaMemcpyHostToDevice
            ),
            "H2D warmup host_b"
        );
    }

    checkCuda(
        cudaDeviceSynchronize(),
        "H2D warmup execution"
    );

    checkCuda(
        cudaEventRecord(start_event),
        "cudaEventRecord H2D start"
    );

    // 正式测量100轮；每一轮都复制a和b两个输入。
    for (int iteration = 0;
         iteration < measured_iterations;
         ++iteration) {
        checkCuda(
            cudaMemcpy(
                device_a,
                host_a.data(),
                byte_count,
                cudaMemcpyHostToDevice
            ),
            "H2D measured host_a"
        );

        checkCuda(
            cudaMemcpy(
                device_b,
                host_b.data(),
                byte_count,
                cudaMemcpyHostToDevice
            ),
            "H2D measured host_b"
        );
    }

    checkCuda(
        cudaEventRecord(stop_event),
        "cudaEventRecord H2D stop"
    );

    checkCuda(
        cudaEventSynchronize(stop_event),
        "cudaEventSynchronize H2D"
    );

    float total_host_to_device_time_ms = 0.0F;

    checkCuda(
        cudaEventElapsedTime(
            &total_host_to_device_time_ms,
            start_event,
            stop_event
        ),
        "cudaEventElapsedTime H2D"
    );

    // 决定GPU线程的组织方式。
    const std::size_t threads_per_block = 256;
    const std::size_t block_count = 256;

    // 预热Kernel，避免首次运行的一次性开销污染正式结果。
    for (int iteration = 0;
         iteration < warmup_iterations;
         ++iteration) {
        vectorAdd<<<block_count, threads_per_block>>>(
            device_a,
            device_b,
            device_c,
            count
        );
    }

    checkCuda(
        cudaGetLastError(),
        "vectorAdd warmup launch"
    );

    checkCuda(
        cudaDeviceSynchronize(),
        "vectorAdd warmup execution"
    );

    // 正式计时开始。
    checkCuda(
        cudaEventRecord(start_event),
        "cudaEventRecord start"
    );

    // 连续运行多次Kernel，减少单次测量的偶然波动。
    for (int iteration = 0;
         iteration < measured_iterations;
         ++iteration) {
        vectorAdd<<<block_count, threads_per_block>>>(
            device_a,
            device_b,
            device_c,
            count
        );
    }

    // 紧跟在最后一次Kernel后记录结束时间。
    checkCuda(
        cudaEventRecord(stop_event),
        "cudaEventRecord stop"
    );

    checkCuda(
        cudaGetLastError(),
        "vectorAdd measured launch"
    );

    checkCuda(
        cudaEventSynchronize(stop_event),
        "vectorAdd measured execution"
    );

    // 计算全部正式迭代的总时间和平均单次时间。
    float total_kernel_time_ms = 0.0F;

    checkCuda(
        cudaEventElapsedTime(
            &total_kernel_time_ms,
            start_event,
            stop_event
        ),
        "cudaEventElapsedTime"
    );

    // 预热GPU到CPU的输出复制，不计入正式结果。
    for (int iteration = 0;
         iteration < warmup_iterations;
         ++iteration) {
        checkCuda(
            cudaMemcpy(
                host_c.data(),
                device_c,
                byte_count,
                cudaMemcpyDeviceToHost
            ),
            "D2H warmup host_c"
        );
    }

    checkCuda(
        cudaDeviceSynchronize(),
        "D2H warmup execution"
    );

    checkCuda(
        cudaEventRecord(start_event),
        "cudaEventRecord D2H start"
    );

    // 正式测量100轮GPU到CPU的输出复制。
    for (int iteration = 0;
         iteration < measured_iterations;
         ++iteration) {
        checkCuda(
            cudaMemcpy(
                host_c.data(),
                device_c,
                byte_count,
                cudaMemcpyDeviceToHost
            ),
            "D2H measured host_c"
        );
    }

    checkCuda(
        cudaEventRecord(stop_event),
        "cudaEventRecord D2H stop"
    );

    checkCuda(
        cudaEventSynchronize(stop_event),
        "cudaEventSynchronize D2H"
    );

    float total_device_to_host_time_ms = 0.0F;

    checkCuda(
        cudaEventElapsedTime(
            &total_device_to_host_time_ms,
            start_event,
            stop_event
        ),
        "cudaEventElapsedTime D2H"
    );

    const int pipeline_warmup_iterations = warmup_iterations;
    const int pipeline_measured_iterations = measured_iterations;

    // 预热完整数据路径，不计入稳定状态端到端时间。
    for (int iteration = 0;
         iteration < pipeline_warmup_iterations;
         ++iteration) {
        checkCuda(
            cudaMemcpy(
                device_a,
                host_a.data(),
                byte_count,
                cudaMemcpyHostToDevice
            ),
            "pipeline warmup H2D host_a"
        );

        checkCuda(
            cudaMemcpy(
                device_b,
                host_b.data(),
                byte_count,
                cudaMemcpyHostToDevice
            ),
            "pipeline warmup H2D host_b"
        );

        vectorAdd<<<block_count, threads_per_block>>>(
            device_a,
            device_b,
            device_c,
            count
        );

        checkCuda(
            cudaGetLastError(),
            "pipeline warmup kernel launch"
        );

        checkCuda(
            cudaMemcpy(
                host_c.data(),
                device_c,
                byte_count,
                cudaMemcpyDeviceToHost
            ),
            "pipeline warmup D2H host_c"
        );
    }

    checkCuda(
        cudaDeviceSynchronize(),
        "pipeline warmup execution"
    );

    // CPU墙钟从输入传输前开始，到输出返回CPU并同步后结束。
    const auto pipeline_start_time =
        std::chrono::steady_clock::now();

    for (int iteration = 0;
         iteration < pipeline_measured_iterations;
         ++iteration) {
        checkCuda(
            cudaMemcpy(
                device_a,
                host_a.data(),
                byte_count,
                cudaMemcpyHostToDevice
            ),
            "pipeline measured H2D host_a"
        );

        checkCuda(
            cudaMemcpy(
                device_b,
                host_b.data(),
                byte_count,
                cudaMemcpyHostToDevice
            ),
            "pipeline measured H2D host_b"
        );

        vectorAdd<<<block_count, threads_per_block>>>(
            device_a,
            device_b,
            device_c,
            count
        );

        checkCuda(
            cudaGetLastError(),
            "pipeline measured kernel launch"
        );

        checkCuda(
            cudaMemcpy(
                host_c.data(),
                device_c,
                byte_count,
                cudaMemcpyDeviceToHost
            ),
            "pipeline measured D2H host_c"
        );
    }

    checkCuda(
        cudaDeviceSynchronize(),
        "pipeline measured execution"
    );

    const auto pipeline_stop_time =
        std::chrono::steady_clock::now();

    const double pipeline_total_time_ms =
        std::chrono::duration<double, std::milli>(
            pipeline_stop_time - pipeline_start_time
        ).count();

    // 注销注册的固定内存
    checkCuda(
        cudaHostUnregister(host_a.data()),
        "cudaHostUnregister host_a"
    );

    checkCuda(
        cudaHostUnregister(host_b.data()),
        "cudaHostUnregister host_b"
    );

    checkCuda(
        cudaHostUnregister(host_c.data()),
        "cudaHostUnregister host_c"
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

    checkCuda(
        cudaEventDestroy(start_event),
        "cudaEventDestroy start"
    );

    checkCuda(
        cudaEventDestroy(stop_event),
        "cudaEventDestroy stop"
    );

    // 自动验证GPU计算结果，避免只凭打印结果人工判断。
    for (std::size_t index = 0; index < count; ++index) {
        const int expected = host_a[index] + host_b[index];

        if (host_c[index] != expected) {
            std::cerr
                << "wrong result at index "
                << index
                << '\n';

            return 1;
        }
    }

    // 先统一完成所有结果计算，然后再集中输出。
    const float average_host_to_device_time_ms =
        total_host_to_device_time_ms /
        static_cast<float>(measured_iterations);

    const float average_kernel_time_ms =
        total_kernel_time_ms /
        static_cast<float>(measured_iterations);

    const float average_device_to_host_time_ms =
        total_device_to_host_time_ms /
        static_cast<float>(measured_iterations);

    const double pipeline_average_time_ms =
        pipeline_total_time_ms /
        static_cast<double>(pipeline_measured_iterations);

    const double host_to_device_gb =
        static_cast<double>(byte_count * 2) /
        1'000'000'000.0;

    const double host_to_device_seconds =
        static_cast<double>(average_host_to_device_time_ms) /
        1000.0;

    const double host_to_device_bandwidth_gbps =
        host_to_device_gb /
        host_to_device_seconds;

    const double device_to_host_gb =
        static_cast<double>(byte_count) /
        1'000'000'000.0;

    const double device_to_host_seconds =
        static_cast<double>(average_device_to_host_time_ms) /
        1000.0;

    const double device_to_host_bandwidth_gb_per_second =
        device_to_host_gb /
        device_to_host_seconds;

    std::cout
        << "element count: " << count << '\n'
        << "threads per block: " << threads_per_block << '\n'
        << "block count: " << block_count << '\n'
        << "launched threads: "
        << block_count * threads_per_block
        << '\n';

    std::cout << "first five results:";

    for (std::size_t index = 0; index < 5; ++index) {
        std::cout << ' ' << host_c[index];
    }

    std::cout << "\nlast five results:";

    for (std::size_t index = count - 5; index < count; ++index) {
        std::cout << ' ' << host_c[index];
    }

    std::cout << '\n';

    std::cout
        << "kernel warmup iterations: "
        << warmup_iterations
        << '\n'
        << "kernel measured iterations: "
        << measured_iterations
        << '\n'
        << "total kernel time: "
        << total_kernel_time_ms
        << " ms\n"
        << "average kernel time: "
        << average_kernel_time_ms
        << " ms\n";

    std::cout
        << "H2D warmup iterations: "
        << warmup_iterations
        << '\n'
        << "H2D measured iterations: "
        << measured_iterations
        << '\n'
        << "H2D total time: "
        << total_host_to_device_time_ms
        << " ms\n"
        << "H2D average time: "
        << average_host_to_device_time_ms
        << " ms\n"
        << "H2D bandwidth: "
        << host_to_device_bandwidth_gbps
        << " GB/s\n"
        << "D2H warmup iterations: "
        << warmup_iterations
        << '\n'
        << "D2H measured iterations: "
        << measured_iterations
        << '\n'
        << "D2H total time: "
        << total_device_to_host_time_ms
        << " ms\n"
        << "D2H average time: "
        << average_device_to_host_time_ms
        << " ms\n"
        << "D2H bandwidth: "
        << device_to_host_bandwidth_gb_per_second
        << " GB/s\n"
        << "pipeline warmup iterations: "
        << pipeline_warmup_iterations
        << '\n'
        << "pipeline measured iterations: "
        << pipeline_measured_iterations
        << '\n'
        << "steady-state pipeline total time: "
        << pipeline_total_time_ms
        << " ms\n"
        << "steady-state pipeline average time: "
        << pipeline_average_time_ms
        << " ms\n";

    return 0;
}
