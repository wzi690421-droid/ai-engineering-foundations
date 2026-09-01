import csv
import time
from pathlib import Path

import torch

from data import create_train_validation_loaders
from model import MLPClassifier, SmallCNN
from train import evaluate, train_one_epoch


def run_experiment(
    model_name,
    model_factory,
    data_dir: Path,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    epochs: int = 5,
    seed: int = 42,
) -> list[dict[str, float | int | str]]:
    # 每组实验都使用可复现的初始化。
    # 两种架构的参数形状不同，因此初始权重不可能逐项相同。
    torch.manual_seed(seed)

    # 每组实验重新创建DataLoader，得到相同的划分和每轮数据顺序。
    train_loader, validation_loader, _, _ = (
        create_train_validation_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            seed=seed,
        )
    )

    # model_factory分别是MLPClassifier类和SmallCNN类。
    # 调用类就会创建一个全新的、尚未训练的模型对象。
    model = model_factory()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    results: list[dict[str, float | int | str]] = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
        )

        validation_loss, validation_accuracy = evaluate(
            model=model,
            data_loader=validation_loader,
        )

        epoch_seconds = time.perf_counter() - epoch_start

        row = {
            "model": model_name,
            "parameter_count": parameter_count,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "epoch_seconds": epoch_seconds,
        }
        results.append(row)

        print(
            f"model={model_name}, "
            f"epoch={epoch}, "
            f"parameters={parameter_count}, "
            f"train_loss={train_loss:.4f}, "
            f"train_accuracy={train_accuracy:.2%}, "
            f"validation_loss={validation_loss:.4f}, "
            f"validation_accuracy={validation_accuracy:.2%}, "
            f"time={epoch_seconds:.2f}s"
        )

    return results


def save_results(
    results: list[dict[str, float | int | str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "model",
        "parameter_count",
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "epoch_seconds",
    ]

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)


def main():
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data"
    output_path = (
        Path(__file__).resolve().parent
        / "results"
        / "model_comparison.csv"
    )

    model_configs = [
        ("mlp", MLPClassifier),
        ("cnn", SmallCNN),
    ]

    all_results: list[dict[str, float | int | str]] = []

    for model_name, model_factory in model_configs:
        experiment_results = run_experiment(
            model_name=model_name,
            model_factory=model_factory,
            data_dir=data_dir,
            batch_size=64,
            learning_rate=0.001,
            epochs=5,
            seed=42,
        )
        all_results.extend(experiment_results)

    save_results(
        results=all_results,
        output_path=output_path,
    )

    print("results saved to:", output_path)


if __name__ == "__main__":
    main()
