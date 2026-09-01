import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch

from data import create_train_validation_loaders
from model import SmallCNN


def parse_args(
    arguments: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测量SmallCNN的PyTorch CPU推理基线。"
    )
    parser.add_argument(
        "--run-id",
        default="run_20260812_165446",
        help="augmentation_on目录中的run编号。",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="正式计时前的预热次数。",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="每个batch size的正式测量次数。",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="PyTorch CPU算子使用的线程数。",
    )
    return parser.parse_args(arguments)


def percentile(
    values: list[float],
    probability: float,
) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return (
        ordered[lower_index] * (1.0 - fraction)
        + ordered[upper_index] * fraction
    )


def benchmark_batch(
    model: SmallCNN,
    images: torch.Tensor,
    warmup: int,
    iterations: int,
) -> dict[str, float | int]:
    # 预热不记录时间，让框架完成首次内存分配和内核准备。
    with torch.inference_mode():
        for _ in range(warmup):
            model(images)

        latencies_ms: list[float] = []
        total_start = time.perf_counter()

        for _ in range(iterations):
            iteration_start = time.perf_counter()
            model(images)
            elapsed_ms = (
                time.perf_counter() - iteration_start
            ) * 1000.0
            latencies_ms.append(elapsed_ms)

        total_seconds = time.perf_counter() - total_start

    batch_size = images.shape[0]
    return {
        "batch_size": batch_size,
        "iterations": iterations,
        "mean_latency_ms": statistics.fmean(latencies_ms),
        "p50_latency_ms": percentile(latencies_ms, 0.50),
        "p95_latency_ms": percentile(latencies_ms, 0.95),
        "throughput_images_per_second": (
            batch_size * iterations / total_seconds
        ),
    }


def write_results(
    output_path: Path,
    rows: list[dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup不能是负数")
    if args.iterations <= 0:
        raise ValueError("--iterations必须是正整数")
    if args.threads <= 0:
        raise ValueError("--threads必须是正整数")

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)

    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parents[2]
    run_dir = (
        current_dir
        / "experiments/small_cnn_augmentation_on"
        / args.run_id
    )
    config_path = run_dir / "config.json"
    best_model_path = run_dir / "checkpoints/best_model.pt"

    if not config_path.is_file():
        raise FileNotFoundError(f"配置不存在：{config_path}")
    if not best_model_path.is_file():
        raise FileNotFoundError(f"最佳模型不存在：{best_model_path}")

    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    _, validation_loader, _, _ = create_train_validation_loaders(
        data_dir=project_root / "data",
        batch_size=64,
        seed=int(config["seed"]),
        validation_size=int(config["validation_size"]),
        use_data_augmentation=False,
    )
    validation_images, _ = next(iter(validation_loader))

    model = SmallCNN()
    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    checkpoint_bytes = best_model_path.stat().st_size

    rows: list[dict[str, object]] = []
    for batch_size in (1, 8, 32, 64):
        result = benchmark_batch(
            model=model,
            images=validation_images[:batch_size],
            warmup=args.warmup,
            iterations=args.iterations,
        )
        result.update(
            {
                "backend": "pytorch_cpu",
                "threads": args.threads,
                "parameter_count": parameter_count,
                "parameter_bytes": parameter_bytes,
                "checkpoint_bytes": checkpoint_bytes,
            }
        )
        rows.append(result)

        print(
            f"batch={batch_size}, "
            f"p50={result['p50_latency_ms']:.4f} ms, "
            f"p95={result['p95_latency_ms']:.4f} ms, "
            f"throughput="
            f"{result['throughput_images_per_second']:.2f} images/s"
        )

    output_path = run_dir / "benchmark/pytorch_cpu.csv"
    write_results(output_path, rows)
    print(f"parameter count: {parameter_count}")
    print(f"parameter size: {parameter_bytes / 1024:.2f} KiB")
    print(f"checkpoint size: {checkpoint_bytes / 1024:.2f} KiB")
    print(f"results saved to: {output_path}")


if __name__ == "__main__":
    main()
