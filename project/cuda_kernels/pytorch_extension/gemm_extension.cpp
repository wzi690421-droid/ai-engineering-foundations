#include <ATen/ATen.h>
#include <torch/library.h>

namespace {

// CPU和CUDA实现共有的输入契约；失败时错误会传回Python。
void checkGemmInputs(
    const at::Tensor& a,
    const at::Tensor& b
) {
    TORCH_CHECK(
        a.dim() == 2 && b.dim() == 2,
        "ai_infra::gemm expects two 2-D tensors"
    );
    TORCH_CHECK(
        a.scalar_type() == at::kFloat &&
            b.scalar_type() == at::kFloat,
        "ai_infra::gemm currently supports float32 only"
    );
    TORCH_CHECK(
        a.device() == b.device(),
        "a and b must be on the same device"
    );
    TORCH_CHECK(
        a.is_contiguous() && b.is_contiguous(),
        "a and b must be contiguous"
    );
    TORCH_CHECK(
        a.size(1) == b.size(0),
        "incompatible matrix shapes: ",
        a.sizes(),
        " and ",
        b.sizes()
    );
}

at::Tensor gemmCpu(
    const at::Tensor& a,
    const at::Tensor& b
) {
    checkGemmInputs(a, b);

    // CPU分支直接调用PyTorch matmul，主要用于接口和结果对齐。
    return at::matmul(a, b);
}

}  // namespace

// CUDA实现在gemm_extension_cuda.cu中；这里先声明供Dispatcher注册。
at::Tensor gemmCuda(
    const at::Tensor& a,
    const at::Tensor& b
);

// 定义算子Schema，Python入口为torch.ops.ai_infra.gemm(a, b)。
TORCH_LIBRARY(ai_infra, library) {
    library.def("gemm(Tensor a, Tensor b) -> Tensor");
}

// 输入是CPU Tensor时，Dispatcher选择gemmCpu。
TORCH_LIBRARY_IMPL(ai_infra, CPU, library) {
    library.impl("gemm", TORCH_FN(gemmCpu));
}

// 输入是CUDA Tensor时，Dispatcher选择gemmCuda。
TORCH_LIBRARY_IMPL(ai_infra, CUDA, library) {
    library.impl("gemm", TORCH_FN(gemmCuda));
}
