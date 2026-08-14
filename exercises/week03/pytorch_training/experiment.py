import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from model import TwoLayerClassifier
from train import evaluate, train_one_epoch

def create_data_loaders(
    data_dir: Path,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    full_dataset = FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # 控制训练集和验证集怎样划分。
    split_generator = torch.Generator().manual_seed(seed)

    train_dataset, validation_dataset = random_split(
        full_dataset,
        lengths=[50_000, 10_000],
        generator=split_generator,
    )

    # 单独控制训练数据每个epoch的打乱顺序。
    shuffle_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, validation_loader

def run_experiment(
    learning_rate: float,
    data_dir: Path,
    epochs: int = 5,
    seed: int = 42,
) -> list[dict[str, float | int]]:
    # 每组实验都重新固定种子，保证模型初始化相同。
    torch.manual_seed(seed)

    # 每组实验都创建新的DataLoader和随机数生成器。
    train_loader, validation_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=64,
        seed=seed,
    )

    model = TwoLayerClassifier(
        input_dim=28 * 28,
        hidden_dim=128,
        num_classes=10,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    results: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
        )

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
        )

        row = {
            "learning_rate": learning_rate,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }

        results.append(row)

        print(
            f"lr={learning_rate}, "
            f"epoch={epoch}, "
            f"train_loss={train_loss:.4f}, "
            f"train_accuracy={train_accuracy:.2%}, "
            f"validation_loss={validation_loss:.4f}, "
            f"validation_accuracy={validation_accuracy:.2%}"
        )

    return results

def save_results(
    results: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "learning_rate",
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
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
        / "learning_rate_comparison.csv"
    )

    all_results: list[dict[str, float | int]] = []

    for learning_rate in (0.01, 0.1):
        experiment_results = run_experiment(
            learning_rate=learning_rate,
            data_dir=data_dir,
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