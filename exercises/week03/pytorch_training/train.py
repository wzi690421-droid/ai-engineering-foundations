import torch
import torch.nn.functional as F

from torch import nn
from pathlib import Path
from model import TwoLayerClassifier
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    # 开启训练模式。
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        # TODO 1：清除上一批留下的梯度
        optimizer.zero_grad()
        # TODO 2：将当前批图片传入模型，得到logits
        logits = model(images)
        # TODO 3：使用logits和labels计算交叉熵
        loss = F.cross_entropy(logits, labels)
        # TODO 4：反向传播
        loss.backward()
        # TODO 5：更新模型参数
        optimizer.step()

        batch_size = labels.shape[0]

        # loss是当前batch的平均损失，乘以batch_size还原成损失总和。
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
    # TODO 1：把模型切换到评估模式
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # TODO 2：关闭自动求导记录
    with torch.no_grad():
        for images, labels in data_loader:
            # TODO 3：前向传播
            logits = model(images)
            # TODO 4：计算交叉熵
            loss = F.cross_entropy(logits,labels)

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy

def save_checkpoint(
    checkpoint_path: Path,
    completed_epochs: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    validation_accuracy: float,
    best_validation_accuracy: float,
) -> None:
    # 如果checkpoints目录不存在，就创建它。
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "completed_epochs": completed_epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "validation_accuracy": validation_accuracy,
        "best_validation_accuracy": best_validation_accuracy,
    }

    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float]:
    # 没有存档时，从第一个epoch开始训练。
    if not checkpoint_path.exists():
        print("checkpoint not found, starting from epoch 1")
        return 0,float("-inf")

    # 读取之前保存的训练状态。
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    best_validation_accuracy = float(
        checkpoint.get(
            "best_validation_accuracy",
            checkpoint["validation_accuracy"],
        )
    )

    # 恢复模型参数。
    model.load_state_dict(
        checkpoint["model_state"]
    )

    # 恢复优化器状态。
    optimizer.load_state_dict(
        checkpoint["optimizer_state"]
    )

    # 取得已经完成的epoch数量。
    completed_epochs = int(
        checkpoint["completed_epochs"]
    )

    print(
        f"checkpoint loaded: "
        f"{completed_epochs} epochs already completed"
    )

    return completed_epochs,best_validation_accuracy

def main():
    torch.manual_seed(42)

    project_root = Path(__file__).resolve().parents[3]
    checkpoint_path = (
        project_root
        / "checkpoints"
        / "latest_checkpoint.pt"
    )
    best_model_path = (
        project_root
        /"checkpoints"
        /"best_model.pt"
    )
    data_dir = project_root / "data"

    full_train_dataset = FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    split_generator = torch.Generator().manual_seed(42)

    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        lengths=[50_000, 10_000],
        generator=split_generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=64,
        shuffle=False,
    )

    print("train samples:", len(train_dataset))
    print("validation samples:", len(validation_dataset))
    print("train steps:", len(train_loader))
    print("validation steps:", len(validation_loader))

    model = TwoLayerClassifier(
        input_dim=28 * 28,
        hidden_dim=128,
        num_classes=10,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    completed_epochs, best_validation_accuracy= load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
    )

    # 每次启动程序时，在已有进度上继续训练5个完整epoch。
    epochs_this_run = 5

    for _ in range(epochs_this_run):
        current_epoch = completed_epochs + 1

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
        )

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
        )

        completed_epochs = current_epoch

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            torch.save(
                model.state_dict(),
                best_model_path,
            )

        save_checkpoint(
            checkpoint_path=checkpoint_path,
            completed_epochs=completed_epochs,
            model=model,
            optimizer=optimizer,
            validation_accuracy=validation_accuracy,
            best_validation_accuracy=best_validation_accuracy,
        )

        print(
            f"epoch {current_epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"train_accuracy={train_accuracy:.2%}, "
            f"validation_loss={validation_loss:.4f}, "
            f"validation_accuracy={validation_accuracy:.2%}"
        )

    reloaded_best_model = TwoLayerClassifier(
        input_dim=28 * 28,
        hidden_dim=128,
        num_classes=10,
    )

    best_model_state = torch.load(
        best_model_path,
        map_location="cpu",
        weights_only=True,
    )

    reloaded_best_model.load_state_dict(
        best_model_state
    )

    best_validation_loss, best_validation_accuracy = evaluate(
        reloaded_best_model,
        validation_loader,
    )

    print(
        f"reloaded best model: "
        f"validation_loss={best_validation_loss:.4f}, "
        f"validation_accuracy={best_validation_accuracy:.2%}"
    )

    print(f"new best model: accuracy={best_validation_accuracy:.2%}")
    print("latest_checkpoint saved to:", checkpoint_path)
    print("best_model saved to", best_model_path)

if __name__ == "__main__":
    main()
