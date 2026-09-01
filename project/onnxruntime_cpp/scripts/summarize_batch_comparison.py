#!/usr/bin/env python3

import argparse
import csv
import statistics
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize C++ ONNX Runtime batch benchmarks."
    )
    parser.add_argument(
        "input_directory",
        type=Path,
        help="Directory containing batch1, batch2, ... result folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to input_directory/comparison.csv.",
    )
    return parser.parse_args()


def read_summary_rows(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))

    if not rows:
        raise ValueError(f"summary contains no runs: {summary_path}")

    return rows


def average_column(rows: list[dict[str, str]], name: str) -> float:
    return statistics.fmean(float(row[name]) for row in rows)


def main() -> None:
    arguments = parse_arguments()
    input_directory = arguments.input_directory.resolve()
    output_path = (
        arguments.output.resolve()
        if arguments.output is not None
        else input_directory / "comparison.csv"
    )

    batch_results: list[dict[str, float | int]] = []

    for batch_directory in input_directory.glob("batch*"):
        batch_text = batch_directory.name.removeprefix("batch")

        if not batch_text.isdigit():
            continue

        batch_size = int(batch_text)
        summary_path = batch_directory / "summary.csv"

        if not summary_path.is_file():
            continue

        rows = read_summary_rows(summary_path)

        average_mean_latency_ms = average_column(
            rows,
            "mean_latency_ms",
        )
        average_p50_latency_ms = average_column(
            rows,
            "p50_latency_ms",
        )
        average_p95_latency_ms = average_column(
            rows,
            "p95_latency_ms",
        )
        average_throughput = average_column(
            rows,
            "throughput_images_per_second",
        )

        batch_results.append(
            {
                "batch_size": batch_size,
                "repeat_count": len(rows),
                "average_mean_latency_ms": average_mean_latency_ms,
                "average_p50_latency_ms": average_p50_latency_ms,
                "average_p95_latency_ms": average_p95_latency_ms,
                "effective_mean_latency_per_image_ms": (
                    average_mean_latency_ms / batch_size
                ),
                "average_throughput_images_per_second": average_throughput,
            }
        )

    batch_results.sort(key=lambda result: int(result["batch_size"]))

    if not batch_results or batch_results[0]["batch_size"] != 1:
        raise ValueError("a batch1 summary is required as the baseline")

    baseline = batch_results[0]
    baseline_throughput = float(
        baseline["average_throughput_images_per_second"]
    )
    baseline_p50 = float(baseline["average_p50_latency_ms"])

    for result in batch_results:
        batch_size = int(result["batch_size"])
        average_p50 = float(result["average_p50_latency_ms"])
        average_throughput = float(
            result["average_throughput_images_per_second"]
        )

        result["throughput_speedup_vs_batch1"] = (
            average_throughput / baseline_throughput
        )
        result["p50_speedup_vs_sequential_batch1"] = (
            batch_size * baseline_p50 / average_p50
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    field_names = list(batch_results[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()

        for result in batch_results:
            formatted_result = {
                name: f"{value:.9f}" if isinstance(value, float) else value
                for name, value in result.items()
            }
            writer.writerow(formatted_result)

    print(f"comparison CSV: {output_path}")


if __name__ == "__main__":
    main()
