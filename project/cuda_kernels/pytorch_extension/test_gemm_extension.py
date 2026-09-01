from pathlib import Path

import torch


ROOT = Path(__file__).parent
LIBRARIES = sorted(ROOT.glob("stage13_gemm_ext*.so"))

# build_ext --inplace后，动态库应当位于当前源码目录。
if len(LIBRARIES) != 1:
    raise RuntimeError(
        "expected exactly one stage13_gemm_ext shared library; "
        "run `python setup.py build_ext --inplace` first"
    )

# 加载.so会执行C++中的TORCH_LIBRARY注册代码。
torch.ops.load_library(str(LIBRARIES[0]))


# FakeTensor实现只推导输出元数据，不执行CPU或CUDA计算。
# torch.compile、FakeTensor和opcheck会使用这一实现。
@torch.library.register_fake("ai_infra::gemm")
def gemm_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    torch._check(a.dim() == 2 and b.dim() == 2)
    torch._check(a.dtype == torch.float32)
    torch._check(b.dtype == torch.float32)
    torch._check(a.device == b.device)
    torch._check(a.is_contiguous() and b.is_contiguous())
    torch._check(a.shape[1] == b.shape[0])
    return a.new_empty((a.shape[0], b.shape[1]))


def assert_matches_reference(
    row_count: int,
    inner_count: int,
    column_count: int,
    device: str,
) -> float:
    # 固定随机种子，让重复运行使用相同输入。
    torch.manual_seed(13)
    a = torch.randn(
        row_count,
        inner_count,
        device=device,
        dtype=torch.float32,
    )
    b = torch.randn(
        inner_count,
        column_count,
        device=device,
        dtype=torch.float32,
    )

    # 自定义算子结果必须与PyTorch参考实现对齐。
    actual = torch.ops.ai_infra.gemm(a, b)
    expected = torch.matmul(a, b)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1.0e-4,
        atol=1.0e-4,
    )
    if actual.numel() == 0:
        return 0.0
    return (actual - expected).abs().max().item()


def measure_cuda_ms(operation, iterations: int = 200) -> float:
    # 先预热，避免首次加载和初始化污染正式计时。
    for _ in range(20):
        operation()

    # CUDA Event在GPU时间线上计时，避免把异步启动误当成执行完成。
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / iterations


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this test requires a CUDA-capable PyTorch build")

    # CPU、非规则CUDA尺寸以及空维度边界的正确性测试。
    cpu_error = assert_matches_reference(37, 29, 41, "cpu")
    cuda_error = assert_matches_reference(513, 509, 517, "cuda")
    assert_matches_reference(5, 0, 7, "cuda")
    assert_matches_reference(0, 3, 4, "cuda")
    assert_matches_reference(4, 3, 0, "cuda")

    # 检查Schema、Autograd注册、FakeTensor和动态形状兼容性。
    opcheck_a = torch.randn(37, 29, device="cuda")
    opcheck_b = torch.randn(29, 41, device="cuda")
    opcheck_result = torch.library.opcheck(
        torch.ops.ai_infra.gemm.default,
        (opcheck_a, opcheck_b),
    )

    # 验证扩展确实使用PyTorch当前Stream，而不是固定默认Stream。
    test_stream = torch.cuda.Stream()
    with torch.cuda.stream(test_stream):
        stream_a = torch.randn(37, 29, device="cuda")
        stream_b = torch.randn(29, 41, device="cuda")
        stream_actual = torch.ops.ai_infra.gemm(
            stream_a,
            stream_b,
        )
        stream_expected = torch.matmul(stream_a, stream_b)
    test_stream.synchronize()
    torch.testing.assert_close(
        stream_actual,
        stream_expected,
        rtol=1.0e-4,
        atol=1.0e-4,
    )

    # 与torch.matmul在相同输入和计时方法下做性能对照。
    benchmark_a = torch.randn(513, 509, device="cuda")
    benchmark_b = torch.randn(509, 517, device="cuda")
    custom_ms = measure_cuda_ms(
        lambda: torch.ops.ai_infra.gemm(
            benchmark_a,
            benchmark_b,
        )
    )
    pytorch_ms = measure_cuda_ms(
        lambda: torch.matmul(benchmark_a, benchmark_b)
    )

    # 当前Kernel不支持任意stride，非连续输入必须明确报错。
    noncontiguous_rejected = False
    try:
        torch.ops.ai_infra.gemm(
            benchmark_a.transpose(0, 1),
            benchmark_b,
        )
    except RuntimeError:
        noncontiguous_rejected = True

    if not noncontiguous_rejected:
        raise AssertionError("non-contiguous input was not rejected")

    print(f"torch: {torch.__version__}")
    print(f"torch CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CPU maximum absolute error: {cpu_error:.8f}")
    print(f"CUDA maximum absolute error: {cuda_error:.8f}")
    print("empty-dimension boundary cases: passed")
    print(f"opcheck: {opcheck_result}")
    print("non-default CUDA stream: passed")
    print("non-contiguous input rejection: passed")
    print(f"custom tiled GEMM: {custom_ms:.3f} ms")
    print(f"torch.matmul: {pytorch_ms:.3f} ms")
    print("PyTorch extension verification passed")


if __name__ == "__main__":
    main()
