import argparse
import csv
import time
import json

from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader

from config import ExperimentConfig
from data import create_train_validation_loaders
from model import SmallCNN


HISTORY_FIELDNAMES = [
    "epoch",
    "train_loss",
    "train_accuracy",
    "validation_loss",
    "validation_accuracy",
    "learning_rate",
    "epoch_seconds",
]


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images,labels in data_loader:
        optimizer.zero_grad()

        logits = model(images)

        loss = F.cross_entropy(logits,labels)

        loss.backward()

        optimizer.step()

        batch_size = labels.shape[0]

        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)

        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy

def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:

            logits = model(images)

            loss = F.cross_entropy(logits,labels)

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


def create_config_record(
    config: ExperimentConfig,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> dict[str, object]:
    if (mean is None) != (std is None):
        raise ValueError("mean和std必须同时提供或同时省略")

    config_data: dict[str, object] = asdict(config)
    if mean is not None and std is not None:
        config_data["normalization"] = {
            "mean": list(mean),
            "std": list(std),
            "estimated_from": "training_split_only",
        }

    return config_data


def update_early_stopping_state(
    validation_accuracy: float,
    early_stopping_reference_accuracy: float,
    epochs_without_improvement: int,
    min_delta: float,
) -> tuple[bool, float, int]:
    significant_improvement = (
        validation_accuracy > early_stopping_reference_accuracy
        and validation_accuracy
        >= early_stopping_reference_accuracy + min_delta
    )

    if significant_improvement:
        return True, validation_accuracy, 0

    return (
        False,
        early_stopping_reference_accuracy,
        epochs_without_improvement + 1,
    )


def save_checkpoint(
    checkpoint_path: Path,
    completed_epochs: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_validation_accuracy: float,
    early_stopping_reference_accuracy: float,
    epochs_without_improvement: int,
    train_generator: torch.Generator,
    config: ExperimentConfig,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> None:
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "completed_epochs": completed_epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_validation_accuracy": best_validation_accuracy,
        "early_stopping_reference_accuracy": (
            early_stopping_reference_accuracy
        ),
        "epochs_without_improvement": epochs_without_improvement,
        # 断点不仅保存训练超参数，也保存真正使用的数据预处理参数。
        "config": create_config_record(config, mean, std),
        # 保存随机状态，保证中断续训时接着使用下一轮数据顺序。
        "train_generator_state": train_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_generator: torch.Generator,
    config: ExperimentConfig,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> tuple[int, float, float, int]:
    if not checkpoint_path.exists():
        print("checkpoint不存在，从Epoch 1开始训练")
        return 0, float("-inf"), float("-inf"), 0

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    saved_config = checkpoint.get("config")

    if saved_config is None:
        raise ValueError(
            "checkpoint没有实验配置，无法确认数据管线是否一致；"
            "请开始一次新的实验"
        )
    else:
        current_config = asdict(config)

        checked_keys = (
            "experiment_name",
            "model_name",
            "seed",
            "batch_size",
            "learning_rate",
            "validation_size",
            "use_data_augmentation",
            "random_crop_padding",
            "horizontal_flip_probability",
            "early_stopping_patience",
            "early_stopping_min_delta",
        )

        for key in checked_keys:
            saved_value = saved_config.get(key)
            if saved_value != current_config[key]:
                raise ValueError(
                    f"checkpoint配置不一致：{key}，"
                    f"旧值={saved_value}，"
                    f"当前值={current_config[key]}"
                )

        saved_normalization = saved_config.get("normalization")
        if saved_normalization is None:
            raise ValueError(
                "checkpoint没有归一化参数，不能在新数据管线上续训；"
                "请开始一次新的实验"
            )

        for name, current_values in (("mean", mean), ("std", std)):
            saved_values = saved_normalization.get(name)
            if saved_values is None or not torch.allclose(
                torch.tensor(saved_values, dtype=torch.float64),
                torch.tensor(current_values, dtype=torch.float64),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"checkpoint归一化参数不一致：{name}，"
                    f"旧值={saved_values}，当前值={list(current_values)}"
                )

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    # 恢复随机状态，让续训的数据顺序与不中断训练保持一致。
    train_generator.set_state(checkpoint["train_generator_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])

    completed_epochs = int(checkpoint["completed_epochs"])
    best_validation_accuracy = float(
        checkpoint["best_validation_accuracy"]
    )
    early_stopping_reference_accuracy = float(
        checkpoint["early_stopping_reference_accuracy"]
    )
    epochs_without_improvement = int(
        checkpoint["epochs_without_improvement"]
    )

    print(
        f"checkpoint已恢复：之前完成了{completed_epochs}个Epoch"
    )

    return (
        completed_epochs,
        best_validation_accuracy,
        early_stopping_reference_accuracy,
        epochs_without_improvement,
    )


def initialize_history(history_path: Path) -> None:
    history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # x模式只允许创建新文件，避免意外覆盖已有训练记录。
    with history_path.open(
        mode="x",
        newline="",
        encoding="utf-8",
    ) as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=HISTORY_FIELDNAMES,
        )
        writer.writeheader()


def append_history(
    history_path: Path,
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    validation_loss: float,
    validation_accuracy: float,
    learning_rate: float,
    epoch_seconds: float,
) -> None:
    if not history_path.is_file():
        raise FileNotFoundError(f"训练历史文件不存在：{history_path}")

    with history_path.open(
        mode="a",
        newline="",
        encoding="utf-8",
    ) as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=HISTORY_FIELDNAMES,
        )

        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rate": learning_rate,
                "epoch_seconds": epoch_seconds,
            }
        )


def plot_history(
    history_path: Path,
    plot_path: Path,
) -> None:
    # 延迟导入：只有真正画图时才加载matplotlib。
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with history_path.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as history_file:
        rows = list(csv.DictReader(history_file))

    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows]
    train_losses = [float(row["train_loss"]) for row in rows]
    validation_losses = [
        float(row["validation_loss"])
        for row in rows
    ]
    train_accuracies = [
        100.0 * float(row["train_accuracy"])
        for row in rows
    ]
    validation_accuracies = [
        100.0 * float(row["validation_accuracy"])
        for row in rows
    ]

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 4.5),
    )

    axes[0].plot(epochs, train_losses, label="train")
    axes[0].plot(epochs, validation_losses, label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_accuracies, label="train")
    axes[1].plot(epochs, validation_accuracies, label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.suptitle(history_path.parents[1].name)
    figure.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)


def parse_args(
    arguments: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练一个新的实验，或者继续已有运行。"
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--new",
        action="store_true",
        help="明确开始一次新实验，并创建新的run时间目录。",
    )
    mode_group.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="在原目录继续指定运行，不创建新run目录。",
    )
    parser.add_argument(
        "--augmentation",
        choices=("on", "off"),
        default="on",
        help="是否对训练集启用随机裁剪和水平翻转，默认on。",
    )
    return parser.parse_args(arguments)


def create_experiment_config(
    augmentation: str,
) -> ExperimentConfig:
    if augmentation not in ("on", "off"):
        raise ValueError("augmentation只能是on或off")

    base_config = ExperimentConfig()
    use_data_augmentation = augmentation == "on"

    # 两组实验自动进入不同目录，避免结果混在一起。
    experiment_name = (
        f"{base_config.model_name}_augmentation_{augmentation}"
    )

    return replace(
        base_config,
        experiment_name=experiment_name,
        use_data_augmentation=use_data_augmentation,
    )


def select_run_dir(
    experiment_root: Path,
    resume_run_id: str | None,
) -> Path:
    if resume_run_id is not None:
        # 只接收目录名，避免误把其他实验目录作为当前实验恢复。
        if Path(resume_run_id).name != resume_run_id:
            raise ValueError("--resume只需要run目录名，不需要完整路径")

        run_dir = experiment_root / resume_run_id
        checkpoint_path = run_dir / "checkpoints" / "latest_checkpoint.pt"

        if not run_dir.is_dir():
            raise FileNotFoundError(f"运行目录不存在：{run_dir}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"运行目录中没有latest checkpoint：{checkpoint_path}"
            )

        return run_dir

    # 默认行为是开始一次独立实验，用启动时间生成运行编号。
    base_run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = experiment_root / base_run_id
    suffix = 1

    # 同一秒启动多个实验时增加序号，防止目录名冲突。
    while run_dir.exists():
        run_dir = experiment_root / f"{base_run_id}_{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True)
    return run_dir

def save_config(
    config_path: Path,
    config: ExperimentConfig,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> None:
    config_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_data = create_config_record(config, mean, std)

    with config_path.open(
        mode="w",
        encoding="utf-8",
    ) as config_file:
        json.dump(
            config_data,
            config_file,
            indent=4,
            ensure_ascii=False,
        )


def main():
    args = parse_args()
    config = create_experiment_config(args.augmentation)

    if config.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience必须是正整数")
    if config.early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta不能是负数")

    torch.manual_seed(config.seed)

    project_root = Path(__file__).resolve().parents[3]

    data_dir = project_root / "data"

    experiment_root = (
        Path(__file__).resolve().parent
        / "experiments"
        / config.experiment_name
    )

    run_dir = select_run_dir(
        experiment_root=experiment_root,
        resume_run_id=args.resume,
    )

    checkpoint_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"

    latest_checkpoint_path = checkpoint_dir / "latest_checkpoint.pt"
    best_model_path = checkpoint_dir / "best_model.pt"
    history_path = results_dir / "history.csv"
    plot_path = results_dir / "training_curves.png"
    config_path = run_dir / "config.json"

    print(f"本次运行目录：{run_dir}")
    if args.resume is None:
        print("运行模式：新实验")
    else:
        print(f"运行模式：续训 {args.resume}")

    if args.resume is None:
        # 新实验先留下配置和CSV表头，再开始读取数据和训练。
        save_config(
            config_path=config_path,
            config=config,
        )
        initialize_history(history_path)

    train_loader, validation_loader, mean, std = (
        create_train_validation_loaders(
            data_dir=data_dir,
            batch_size=config.batch_size,
            seed=config.seed,
            validation_size=config.validation_size,
            use_data_augmentation=config.use_data_augmentation,
            random_crop_padding=config.random_crop_padding,
            horizontal_flip_probability=(
                config.horizontal_flip_probability
            ),
        )
    )

    # data.py为训练DataLoader显式创建了这个生成器。
    train_generator = train_loader.generator
    if train_generator is None:
        raise RuntimeError("训练DataLoader缺少随机数生成器")

    model = SmallCNN()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    (
        completed_epochs,
        best_validation_accuracy,
        early_stopping_reference_accuracy,
        epochs_without_improvement,
    ) = load_checkpoint(
        checkpoint_path=latest_checkpoint_path,
        model=model,
        optimizer=optimizer,
        train_generator=train_generator,
        config=config,
        mean=mean,
        std=std,
    )

    if args.resume is not None:
        # load_checkpoint已经验证配置；续训不得创建新的历史文件。
        if not history_path.is_file():
            raise FileNotFoundError(
                f"续训运行缺少历史文件：{history_path}"
            )

    # 新实验在统计完成后补充派生参数；续训只在配置校验通过后更新。
    save_config(
        config_path=config_path,
        config=config,
        mean=mean,
        std=std,
    )

    remaining_epochs = max(
        config.target_epochs - completed_epochs,
        0,
    )

    print(
        f"目标轮数：{config.target_epochs}，"
        f"已完成：{completed_epochs}，"
        f"本次还需训练：{remaining_epochs}"
    )


    for _ in range(remaining_epochs):
        current_epoch = completed_epochs + 1
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
        current_learning_rate = optimizer.param_groups[0]["lr"]
        completed_epochs = current_epoch

        is_new_best = (
            validation_accuracy > best_validation_accuracy
        )
        if is_new_best:
            best_validation_accuracy = validation_accuracy
            best_model_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            torch.save(model.state_dict(), best_model_path)

        (
            significant_improvement,
            early_stopping_reference_accuracy,
            epochs_without_improvement,
        ) = update_early_stopping_state(
            validation_accuracy=validation_accuracy,
            early_stopping_reference_accuracy=(
                early_stopping_reference_accuracy
            ),
            epochs_without_improvement=epochs_without_improvement,
            min_delta=config.early_stopping_min_delta,
        )

        save_checkpoint(
            checkpoint_path=latest_checkpoint_path,
            completed_epochs=completed_epochs,
            model=model,
            optimizer=optimizer,
            best_validation_accuracy=best_validation_accuracy,
            early_stopping_reference_accuracy=(
                early_stopping_reference_accuracy
            ),
            epochs_without_improvement=epochs_without_improvement,
            train_generator=train_generator,
            config=config,
            mean=mean,
            std=std,
        )

        append_history(
            history_path=history_path,
            epoch=current_epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
            learning_rate=current_learning_rate,
            epoch_seconds=epoch_seconds,
        )

        # 曲线由CSV重新读取生成，因此CSV始终是原始数据来源。
        plot_history(
            history_path=history_path,
            plot_path=plot_path,
        )

        print(
            f"epoch={current_epoch}, "
            f"train_loss={train_loss:.4f}, "
            f"train_accuracy={train_accuracy:.2%}, "
            f"validation_loss={validation_loss:.4f}, "
            f"validation_accuracy={validation_accuracy:.2%}, "
            f"significant_improvement={significant_improvement}, "
            f"patience={epochs_without_improvement}/"
            f"{config.early_stopping_patience}, "
            f"time={epoch_seconds:.2f}s"
        )

        if (
            epochs_without_improvement
            >= config.early_stopping_patience
        ):
            print(
                "触发Early Stopping：验证准确率连续"
                f"{epochs_without_improvement}轮没有提升"
            )
            print(
                f"最佳验证准确率={best_validation_accuracy:.2%}，"
                f"最佳模型={best_model_path}"
            )
            break


if __name__ == "__main__":
    main()
