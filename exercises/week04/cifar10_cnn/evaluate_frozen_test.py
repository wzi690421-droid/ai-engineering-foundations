import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from data import create_test_loader
from model import SmallCNN


FROZEN_RUN_ID = "run_20260812_165446"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_best_validation_row(history_path: Path) -> dict[str, str]:
    with history_path.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as history_file:
        rows = list(csv.DictReader(history_file))
    return max(
        rows,
        key=lambda row: float(row["validation_accuracy"]),
    )


def main() -> None:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parents[2]
    run_dir = (
        current_dir
        / "experiments/small_cnn_augmentation_on"
        / FROZEN_RUN_ID
    )
    evidence_dir = current_dir / "stage05_evidence"
    manifest_path = evidence_dir / "frozen_model_manifest.json"
    metrics_path = evidence_dir / "final_test_metrics.json"

    # 测试结果一旦存在就拒绝重复评估，避免利用测试集继续调参。
    if metrics_path.exists():
        raise FileExistsError(
            f"最终测试已经执行过，拒绝重复运行：{metrics_path}"
        )

    config_path = run_dir / "config.json"
    history_path = run_dir / "results/history.csv"
    best_model_path = run_dir / "checkpoints/best_model.pt"

    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    best_validation_row = find_best_validation_row(history_path)
    model_sha256 = sha256_file(best_model_path)

    manifest = {
        "run_id": FROZEN_RUN_ID,
        "model_path": str(best_model_path),
        "model_sha256": model_sha256,
        "selection_source": "validation_set_only",
        "selection_metric": "validation_accuracy",
        "best_validation_epoch": int(best_validation_row["epoch"]),
        "best_validation_accuracy": float(
            best_validation_row["validation_accuracy"]
        ),
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=4),
        encoding="utf-8",
    )

    normalization = config["normalization"]
    test_loader = create_test_loader(
        data_dir=project_root / "data",
        mean=tuple(normalization["mean"]),
        std=tuple(normalization["std"]),
        batch_size=int(config["batch_size"]),
    )

    model = SmallCNN()
    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()

    class_names = test_loader.dataset.classes
    confusion_matrix = torch.zeros((10, 10), dtype=torch.int64)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            predictions = logits.argmax(dim=1)
            batch_size = labels.shape[0]

            total_loss += float(loss.item()) * batch_size
            total_correct += int((predictions == labels).sum().item())
            total_samples += batch_size

            for label, prediction in zip(labels, predictions):
                confusion_matrix[
                    int(label.item()),
                    int(prediction.item()),
                ] += 1

    per_class_accuracy = {}
    for class_index, class_name in enumerate(class_names):
        class_total = int(confusion_matrix[class_index].sum().item())
        class_correct = int(
            confusion_matrix[class_index, class_index].item()
        )
        per_class_accuracy[class_name] = class_correct / class_total

    metrics = {
        "evaluated_at": datetime.now().astimezone().isoformat(),
        "run_id": FROZEN_RUN_ID,
        "model_sha256": model_sha256,
        "test_sample_count": total_samples,
        "test_loss": total_loss / total_samples,
        "test_accuracy": total_correct / total_samples,
        "per_class_accuracy": per_class_accuracy,
        "policy": (
            "one final evaluation after model and configuration freeze; "
            "do not tune from these test metrics"
        ),
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    print(f"frozen run: {FROZEN_RUN_ID}")
    print(
        "best validation: "
        f"{manifest['best_validation_accuracy']:.2%} "
        f"at epoch {manifest['best_validation_epoch']}"
    )
    print(f"test loss: {metrics['test_loss']:.4f}")
    print(f"test accuracy: {metrics['test_accuracy']:.2%}")
    print(f"manifest: {manifest_path}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
