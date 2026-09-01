#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr unsigned int kTileSize = 16;

__global__ void gemmTiledKernel(
    // A、B、C都是PyTorch Tensor底层显存的裸指针。
    const float* a,
    const float* b,
    float* c,
    // 矩阵形状：A[M, K] × B[K, N] = C[M, N]。
    const std::int64_t row_count,
    const std::int64_t inner_count,
    const std::int64_t column_count
) {
    // 每个Block拥有独立的两块Shared Memory，分别缓存一个A Tile和B Tile。
    __shared__ float shared_a[kTileSize][kTileSize];
    __shared__ float shared_b[kTileSize][kTileSize];

    // 当前线程在16×16 Tile内部负责的列和行。
    const unsigned int local_column = threadIdx.x;
    const unsigned int local_row = threadIdx.y;

    // 当前线程最终负责计算的全局输出坐标C[row, column]。
    const std::int64_t column =
        static_cast<std::int64_t>(blockIdx.x) *
        kTileSize +
        local_column;
    const std::int64_t row =
        static_cast<std::int64_t>(blockIdx.y) *
        kTileSize +
        local_row;

    // 每个线程用自己的寄存器累加一个C元素。
    float sum = 0.0F;

    // 沿K维每次前进16个元素，依次处理所有A、B Tile。
    for (std::int64_t tile_start = 0;
         tile_start < inner_count;
         tile_start += kTileSize) {
        // 当前线程负责搬运的A列坐标和B行坐标。
        const std::int64_t a_column =
            tile_start + local_column;
        const std::int64_t b_row =
            tile_start + local_row;

        // 256个线程协作加载A Tile；越界位置补0。
        shared_a[local_row][local_column] =
            row < row_count && a_column < inner_count
                ? a[row * inner_count + a_column]
                : 0.0F;

        // 256个线程协作加载B Tile；越界位置补0。
        shared_b[local_row][local_column] =
            b_row < inner_count && column < column_count
                ? b[b_row * column_count + column]
                : 0.0F;

        // 等待整个Block把当前两个Tile完整地放入Shared Memory。
        __syncthreads();

        // 当前线程读取Shared Memory中的一行A和一列B，完成16次乘加。
        #pragma unroll
        for (unsigned int inner = 0;
             inner < kTileSize;
             ++inner) {
            sum +=
                shared_a[local_row][inner] *
                shared_b[inner][local_column];
        }

        // 等待所有线程用完当前Tile，再允许下一轮覆盖Shared Memory。
        __syncthreads();
    }

    // 边缘Block中的越界线程不写C，但此前仍参与加载和同步。
    if (row < row_count && column < column_count) {
        c[row * column_count + column] = sum;
    }
}

void checkCudaInputs(
    const at::Tensor& a,
    const at::Tensor& b
) {
    // Dispatcher虽然已经按CUDA设备分发，但这里仍保留明确的防御性检查。
    TORCH_CHECK(
        a.is_cuda() && b.is_cuda(),
        "CUDA implementation expects CUDA tensors"
    );

    // 当前Kernel只实现普通二维矩阵乘法。
    TORCH_CHECK(
        a.dim() == 2 && b.dim() == 2,
        "ai_infra::gemm expects two 2-D tensors"
    );

    // Kernel内部使用float*，因此目前只接受FP32 Tensor。
    TORCH_CHECK(
        a.scalar_type() == at::kFloat &&
            b.scalar_type() == at::kFloat,
        "ai_infra::gemm currently supports float32 only"
    );

    // 两个输入必须位于同一张GPU，否则裸指针不能由同一次Kernel正确访问。
    TORCH_CHECK(
        a.device() == b.device(),
        "a and b must be on the same device"
    );

    // Kernel按行优先连续下标访问，暂不处理任意stride。
    TORCH_CHECK(
        a.is_contiguous() && b.is_contiguous(),
        "a and b must be contiguous"
    );

    // 矩阵乘法要求A[M, K]的K等于B[K, N]的K。
    TORCH_CHECK(
        a.size(1) == b.size(0),
        "incompatible matrix shapes: ",
        a.sizes(),
        " and ",
        b.sizes()
    );
}

}  // namespace

at::Tensor gemmCuda(
    const at::Tensor& a,
    const at::Tensor& b
) {
    // 在访问裸指针和启动Kernel前先验证完整输入契约。
    checkCudaInputs(a, b);

    // 从Tensor元数据得到M、K、N，不需要Python额外传入尺寸。
    const auto row_count = a.size(0);
    const auto inner_count = a.size(1);
    const auto column_count = b.size(1);

    // 由PyTorch分配输出显存；C继承A的device和dtype，形状为[M, N]。
    auto c = at::empty(
        {row_count, column_count},
        a.options()
    );

    // CUDA不允许启动某一维为0的Grid；空输出可以直接返回。
    if (row_count == 0 || column_count == 0) {
        return c;
    }

    // 临时切换到输入Tensor所在GPU，函数返回时自动恢复原设备。
    const c10::cuda::CUDAGuard device_guard(a.device());

    // 使用PyTorch当前CUDA Stream，保持与前后PyTorch算子的执行顺序一致。
    const auto stream =
        c10::cuda::getCurrentCUDAStream(a.get_device());

    // 每个Block包含16×16个线程，每个线程计算一个C元素。
    const dim3 block(kTileSize, kTileSize);

    // 向上取整得到覆盖整个[M, N]输出矩阵所需的Block数量。
    const dim3 grid(
        static_cast<unsigned int>(
            (column_count + kTileSize - 1) /
            kTileSize
        ),
        static_cast<unsigned int>(
            (row_count + kTileSize - 1) /
            kTileSize
        )
    );

    // Tensor不发生复制，只把已有显存的裸指针交给CUDA Kernel。
    gemmTiledKernel<<<grid, block, 0, stream.stream()>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        row_count,
        inner_count,
        column_count
    );

    // 只检查Kernel启动错误，不在这里强制等待GPU完成计算。
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // 返回的Tensor仍由PyTorch管理显存与生命周期。
    return c;
}
