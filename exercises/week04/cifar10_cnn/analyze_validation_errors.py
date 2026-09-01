import argparse
import csv
import json
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import create_train_validation_loaders
from model import SmallCNN


def parse_args(
    arguments: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分析最佳模型在固定验证集上的预测错误。"
    )
    parser.add_argument(
        "--augmentation",
        choices=("on", "off"),
        default="on",
        help="选择要分析的数据增强实验，默认on。",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="指定run目录名；省略时自动选择最近的运行。",
    )
    parser.add_argument(
        "--high-confidence-limit",
        type=int,
        default=50,
        help="保存置信度最高的错误数量，默认50。",
    )
    return parser.parse_args(arguments)


def select_run_dir(
    experiments_dir: Path,
    augmentation: str,
    run_id: str | None,
) -> Path:
    experiment_dir = (
        experiments_dir
        / f"small_cnn_augmentation_{augmentation}"
    )

    if run_id is not None:
        if Path(run_id).name != run_id:
            raise ValueError("--run-id只需要目录名，不需要完整路径")
        run_dir = experiment_dir / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"运行目录不存在：{run_dir}")
        return run_dir

    candidates = [
        path
        for path in experiment_dir.glob("run_*")
        if (path / "checkpoints/best_model.pt").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"没有找到augmentation={augmentation}的最佳模型"
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        return json.load(config_file)


def collect_predictions(
    model: SmallCNN,
    validation_loader,
) -> tuple[list[dict[str, object]], torch.Tensor]:
    class_names = validation_loader.dataset.dataset.classes
    original_indices = validation_loader.dataset.indices
    class_count = len(class_names)
    confusion_matrix = torch.zeros(
        (class_count, class_count),
        dtype=torch.int64,
    )
    prediction_rows: list[dict[str, object]] = []
    validation_position = 0

    model.eval()
    with torch.no_grad():
        for images, labels in validation_loader:
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)

            for label, prediction, confidence in zip(
                labels,
                predictions,
                confidences,
            ):
                true_index = int(label.item())
                predicted_index = int(prediction.item())
                confusion_matrix[true_index, predicted_index] += 1

                prediction_rows.append(
                    {
                        "validation_position": validation_position,
                        "original_dataset_index": original_indices[
                            validation_position
                        ],
                        "true_index": true_index,
                        "true_class": class_names[true_index],
                        "predicted_index": predicted_index,
                        "predicted_class": class_names[predicted_index],
                        "confidence": float(confidence.item()),
                        "correct": true_index == predicted_index,
                    }
                )
                validation_position += 1

    return prediction_rows, confusion_matrix


def write_csv(
    output_path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        raise ValueError("没有可写入的分析结果")

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


def create_per_class_rows(
    confusion_matrix: torch.Tensor,
    class_names: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_index, class_name in enumerate(class_names):
        total = int(confusion_matrix[class_index].sum().item())
        correct = int(confusion_matrix[class_index, class_index].item())
        rows.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "correct": correct,
                "total": total,
                "accuracy": correct / total,
            }
        )
    return rows


def plot_confusion_matrix(
    confusion_matrix: torch.Tensor,
    class_names: list[str],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(confusion_matrix.numpy(), cmap="Blues")
    figure.colorbar(image, ax=axis)

    axis.set(
        title="Validation Confusion Matrix",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_high_confidence_errors(
    error_rows,
    raw_dataset,
    output_path: Path,
    limit: int = 10,
) -> None:
    # error_rows已经按照置信度从高到低排列。
    # [:limit]只选择最自信的前limit个错误。
    selected_errors = error_rows[:limit]

    figure, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(15, 6),
    )

    # axes原本是2×5的二维结构。
    # flat把它展开，方便后面逐个取出子图。
    for axis in axes.flat:
        axis.axis("off")


    for axis, error in zip(axes.flat, selected_errors):
        original_index = int(error["original_dataset_index"])
        image, _ = raw_dataset[original_index]
        confidence = float(error["confidence"])

        axis.imshow(image)
        axis.set_title(
            f"true: {error['true_class']}\n"
            f"pred: {error['predicted_class']}\n"
            f"confidence: {confidence:.1%}"
        )

    figure.suptitle("Highest-confidence Validation Errors")
    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.94),
        h_pad=2.0,
        w_pad=1.5,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.high_confidence_limit <= 0:
        raise ValueError("--high-confidence-limit必须是正整数")

    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parents[2]
    run_dir = select_run_dir(
        experiments_dir=current_dir / "experiments",
        augmentation=args.augmentation,
        run_id=args.run_id,
    )
    config = load_run_config(run_dir)

    _, validation_loader, mean, std = (
        create_train_validation_loaders(
            data_dir=project_root / "data",
            batch_size=int(config["batch_size"]),
            seed=int(config["seed"]),
            validation_size=int(config["validation_size"]),
            use_data_augmentation=bool(
                config["use_data_augmentation"]
            ),
            random_crop_padding=int(config["random_crop_padding"]),
            horizontal_flip_probability=float(
                config["horizontal_flip_probability"]
            ),
        )
    )

    saved_normalization = config["normalization"]
    if list(mean) != saved_normalization["mean"]:
        raise ValueError("重新计算的mean与实验配置不一致")
    if list(std) != saved_normalization["std"]:
        raise ValueError("重新计算的std与实验配置不一致")

    model = SmallCNN()
    model.load_state_dict(
        torch.load(
            run_dir / "checkpoints/best_model.pt",
            map_location="cpu",
            weights_only=True,
        )
    )

    prediction_rows, confusion_matrix = collect_predictions(
        model=model,
        validation_loader=validation_loader,
    )
    class_names = validation_loader.dataset.dataset.classes
    per_class_rows = create_per_class_rows(
        confusion_matrix=confusion_matrix,
        class_names=class_names,
    )
    error_rows = sorted(
        (
            row
            for row in prediction_rows
            if not row["correct"]
        ),
        key=lambda row: float(row["confidence"]),
        reverse=True,
    )

    analysis_dir = run_dir / "analysis"
    write_csv(
        analysis_dir / "validation_predictions.csv",
        prediction_rows,
    )
    write_csv(
        analysis_dir / "per_class_accuracy.csv",
        per_class_rows,
    )
    write_csv(
        analysis_dir / "high_confidence_errors.csv",
        error_rows[: args.high_confidence_limit],
    )
    plot_confusion_matrix(
        confusion_matrix=confusion_matrix,
        class_names=class_names,
        output_path=analysis_dir / "confusion_matrix.png",
    )
    plot_high_confidence_errors(
        error_rows=error_rows,
        raw_dataset=validation_loader.dataset.dataset,
        output_path=(
            analysis_dir / "high_confidence_errors.png"
        ),
        limit=10,
    )

    correct_count = sum(
        int(bool(row["correct"]))
        for row in prediction_rows
    )
    print(f"run: {run_dir}")
    print(
        f"validation accuracy: "
        f"{correct_count / len(prediction_rows):.2%}"
    )
    for row in sorted(
        per_class_rows,
        key=lambda item: float(item["accuracy"]),
    ):
        print(
            f"{row['class_name']}: "
            f"{float(row['accuracy']):.2%}"
        )
    print(f"analysis saved to: {analysis_dir}")


if __name__ == "__main__":
    main()
