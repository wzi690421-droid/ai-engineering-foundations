#!/usr/bin/env python3

import argparse
import csv
import math
import statistics
from pathlib import Path


STAGES = (
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "end_to_end_ms",
)


def percentile(values: list[float], ratio: float) -> float:
    sorted_values = sorted(values)
    position = ratio * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return sorted_values[lower_index]

    upper_weight = position - lower_index
    lower_weight = 1.0 - upper_weight

    return (
        sorted_values[lower_index] * lower_weight
        + sorted_values[upper_index] * upper_weight
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize repeated pipeline benchmark CSV files."
    )
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    input_directory = arguments.input_directory.resolve()
    output_path = (
        arguments.output.resolve()
        if arguments.output is not None
        else input_directory / "summary.csv"
    )

    run_files = sorted(input_directory.glob("run_*.csv"))

    if not run_files:
        raise ValueError(f"no run CSV files found in {input_directory}")

    summary_rows: list[dict[str, int | float]] = []

    for run_index, run_path in enumerate(run_files, start=1):
        with run_path.open(newline="", encoding="utf-8") as input_file:
            rows = list(csv.DictReader(input_file))

        if not rows:
            raise ValueError(f"pipeline CSV contains no samples: {run_path}")

        summary: dict[str, int | float] = {
            "run": run_index,
            "batch_size": int(rows[0]["batch_size"]),
            "warmup_iterations": int(rows[0]["warmup_iterations"]),
            "measured_iterations": int(rows[0]["measured_iterations"]),
        }

        for stage in STAGES:
            values = [float(row[stage]) for row in rows]
            stage_name = stage.removesuffix("_ms")
            summary[f"{stage_name}_mean_ms"] = statistics.fmean(values)
            summary[f"{stage_name}_p50_ms"] = percentile(values, 0.50)
            summary[f"{stage_name}_p95_ms"] = percentile(values, 0.95)

        summary_rows.append(summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        field_names = list(summary_rows[0].keys())
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()

        for summary in summary_rows:
            writer.writerow(
                {
                    name: f"{value:.9f}" if isinstance(value, float) else value
                    for name, value in summary.items()
                }
            )

    print(f"pipeline summary CSV: {output_path}")


if __name__ == "__main__":
    main()
