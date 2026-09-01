import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_latest_history(
    experiments_dir: Path,
    augmentation: str,
) -> Path:
    experiment_dir = (
        experiments_dir
        / f"small_cnn_augmentation_{augmentation}"
    )
    history_paths = list(
        experiment_dir.glob("run_*/results/history.csv")
    )

    if not history_paths:
        raise FileNotFoundError(
            f"没有找到augmentation={augmentation}的history.csv"
        )

    # 用文件修改时间找到最近完成或更新的那次运行。
    return max(
        history_paths,
        key=lambda path: path.stat().st_mtime,
    )


def read_history(history_path: Path) -> list[dict[str, float]]:
    with history_path.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as history_file:
        reader = csv.DictReader(history_file)
        return [
            {
                "epoch": float(row["epoch"]),
                "validation_loss": float(row["validation_loss"]),
                "validation_accuracy": float(
                    row["validation_accuracy"]
                ),
            }
            for row in reader
        ]


def plot_comparison(
    histories: dict[str, list[dict[str, float]]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 4.5),
    )

    for augmentation, rows in histories.items():
        label = f"augmentation {augmentation}"
        epochs = [row["epoch"] for row in rows]
        validation_losses = [
            row["validation_loss"]
            for row in rows
        ]
        validation_accuracies = [
            100.0 * row["validation_accuracy"]
            for row in rows
        ]

        axes[0].plot(
            epochs,
            validation_losses,
            label=label,
        )
        axes[1].plot(
            epochs,
            validation_accuracies,
            label=label,
        )

    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.suptitle("Data Augmentation A/B Comparison")
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def print_summary(
    histories: dict[str, list[dict[str, float]]],
) -> None:
    best_accuracies: dict[str, float] = {}

    for augmentation, rows in histories.items():
        best_accuracy_row = max(
            rows,
            key=lambda row: row["validation_accuracy"],
        )
        best_loss_row = min(
            rows,
            key=lambda row: row["validation_loss"],
        )
        best_accuracies[augmentation] = best_accuracy_row[
            "validation_accuracy"
        ]

        print(
            f"augmentation={augmentation}: "
            f"best_accuracy="
            f"{best_accuracy_row['validation_accuracy']:.2%} "
            f"(epoch {int(best_accuracy_row['epoch'])}), "
            f"best_loss={best_loss_row['validation_loss']:.4f} "
            f"(epoch {int(best_loss_row['epoch'])})"
        )

    improvement = (
        best_accuracies["on"]
        - best_accuracies["off"]
    )
    print(
        "best accuracy improvement: "
        f"{100.0 * improvement:.2f} percentage points"
    )


def main() -> None:
    current_dir = Path(__file__).resolve().parent
    experiments_dir = current_dir / "experiments"

    history_paths = {
        augmentation: find_latest_history(
            experiments_dir=experiments_dir,
            augmentation=augmentation,
        )
        for augmentation in ("off", "on")
    }
    histories = {
        augmentation: read_history(history_path)
        for augmentation, history_path in history_paths.items()
    }

    for augmentation, history_path in history_paths.items():
        print(f"augmentation={augmentation}: {history_path}")

    output_path = (
        experiments_dir
        / "augmentation_comparison.png"
    )
    plot_comparison(histories, output_path)
    print_summary(histories)
    print(f"comparison saved to: {output_path}")


if __name__ == "__main__":
    main()
