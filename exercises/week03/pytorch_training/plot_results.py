import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(
    csv_path: Path,
) -> dict[float, dict[str, list[float]]]:
    grouped_results: dict[
        float,
        dict[str, list[float]],
    ] = {}

    with csv_path.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            learning_rate = float(row["learning_rate"])

            if learning_rate not in grouped_results:
                grouped_results[learning_rate] = {
                    "epoch": [],
                    "validation_loss": [],
                    "validation_accuracy": [],
                }

            grouped_results[learning_rate]["epoch"].append(
                float(row["epoch"])
            )
            grouped_results[learning_rate]["validation_loss"].append(
                float(row["validation_loss"])
            )
            grouped_results[learning_rate]["validation_accuracy"].append(
                float(row["validation_accuracy"]) * 100
            )

    return grouped_results


def plot_results(
    grouped_results: dict[
        float,
        dict[str, list[float]],
    ],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 4),
    )

    for learning_rate in sorted(grouped_results):
        result = grouped_results[learning_rate]

        axes[0].plot(
            result["epoch"],
            result["validation_loss"],
            marker="o",
            label=f"lr={learning_rate}",
        )

        axes[1].plot(
            result["epoch"],
            result["validation_accuracy"],
            marker="o",
            label=f"lr={learning_rate}",
        )

    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True)
    axes[1].legend()

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=160,
    )

    plt.close(figure)


def main():
    results_dir = Path(__file__).resolve().parent / "results"

    csv_path = results_dir / "learning_rate_comparison.csv"
    output_path = results_dir / "learning_rate_comparison.png"

    grouped_results = load_results(csv_path)

    plot_results(
        grouped_results,
        output_path,
    )

    print("plot saved to:", output_path)


if __name__ == "__main__":
    main()