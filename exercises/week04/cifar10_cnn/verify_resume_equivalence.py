import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from data import TransformedSubset, create_training_transform
from model import SmallCNN
from train import (
    create_experiment_config,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)


SEED = 42
SAMPLE_COUNT = 256
BATCH_SIZE = 64
MEAN = (
    0.49165572255588375,
    0.48233405909578414,
    0.44667924095728523,
)
STD = (
    0.24708116027851917,
    0.24350869632987013,
    0.2615806648572366,
)


def create_training_objects(data_dir: Path):
    raw_dataset = CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=None,
    )
    dataset = TransformedSubset(
        dataset=raw_dataset,
        indices=list(range(SAMPLE_COUNT)),
        transform=create_training_transform(MEAN, STD),
    )
    generator = torch.Generator().manual_seed(SEED)
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    model = SmallCNN()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )
    return data_loader, generator, model, optimizer


def clone_model_state(model: SmallCNN) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


def run_continuously(data_dir: Path):
    torch.manual_seed(SEED)
    data_loader, _, model, optimizer = create_training_objects(data_dir)
    metrics = [
        train_one_epoch(model, data_loader, optimizer)
        for _ in range(2)
    ]
    return clone_model_state(model), metrics


def run_with_restart(data_dir: Path, checkpoint_path: Path):
    config = create_experiment_config("on")

    # 第一段进程：训练一轮并保存全部续训状态。
    torch.manual_seed(SEED)
    data_loader, generator, model, optimizer = (
        create_training_objects(data_dir)
    )
    first_metrics = train_one_epoch(model, data_loader, optimizer)
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        completed_epochs=1,
        model=model,
        optimizer=optimizer,
        best_validation_accuracy=0.0,
        early_stopping_reference_accuracy=0.0,
        epochs_without_improvement=0,
        train_generator=generator,
        config=config,
        mean=MEAN,
        std=STD,
    )

    # 第二段进程：故意使用不同种子重新创建对象，随后由checkpoint恢复。
    torch.manual_seed(999)
    resumed_loader, resumed_generator, resumed_model, resumed_optimizer = (
        create_training_objects(data_dir)
    )
    restored_state = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        train_generator=resumed_generator,
        config=config,
        mean=MEAN,
        std=STD,
    )
    if restored_state != (1, 0.0, 0.0, 0):
        raise AssertionError(f"恢复状态不正确：{restored_state}")

    second_metrics = train_one_epoch(
        resumed_model,
        resumed_loader,
        resumed_optimizer,
    )
    return (
        clone_model_state(resumed_model),
        [first_metrics, second_metrics],
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    evidence_path = (
        Path(__file__).resolve().parent
        / "stage05_evidence/resume_equivalence.json"
    )

    continuous_state, continuous_metrics = run_continuously(
        project_root / "data"
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        resumed_state, resumed_metrics = run_with_restart(
            data_dir=project_root / "data",
            checkpoint_path=(
                Path(temporary_directory) / "checkpoint.pt"
            ),
        )

    maximum_parameter_difference = max(
        float(
            (continuous_state[name] - resumed_state[name])
            .abs()
            .max()
            .item()
        )
        for name in continuous_state
    )
    metrics_equal = continuous_metrics == resumed_metrics
    parameters_equal = all(
        torch.equal(continuous_state[name], resumed_state[name])
        for name in continuous_state
    )

    report = {
        "sample_count": SAMPLE_COUNT,
        "batch_size": BATCH_SIZE,
        "continuous_metrics": continuous_metrics,
        "resumed_metrics": resumed_metrics,
        "metrics_exactly_equal": metrics_equal,
        "parameters_exactly_equal": parameters_equal,
        "maximum_parameter_difference": maximum_parameter_difference,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    if not metrics_equal or not parameters_equal:
        raise AssertionError(
            "续训与连续训练不等价，请查看resume_equivalence.json"
        )

    print("resume equivalence verification passed")
    print(f"maximum parameter difference: {maximum_parameter_difference}")
    print(f"evidence saved to: {evidence_path}")


if __name__ == "__main__":
    main()
