import torch
import torch.nn.functional as F

from model import TwoLayerClassifier
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor


def main():
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data"

    # Dataset 负责表示整个数据集，以及如何读取一个样本。
    train_dataset = FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # DataLoader 负责分批、打乱以及依次取出数据。
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )

    # iter() 创建批次迭代器，next() 取出第一个批次。
    images, labels = next(iter(train_loader))

    print("dataset size:", len(train_dataset))
    print("images.shape:", images.shape)
    print("labels.shape:", labels.shape)
    print("images.dtype:", images.dtype)
    print("labels.dtype:", labels.dtype)
    print("pixel range:", images.min().item(), images.max().item())

    ordered_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=False,
    )

    shuffled_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )

    _, ordered_labels = next(iter(ordered_loader))
    _, shuffled_labels = next(iter(shuffled_loader))

    print("ordered labels:", ordered_labels.tolist())
    print("shuffled labels:", shuffled_labels.tolist())
    print("steps per epoch:", len(train_loader))

    # MLP不能直接接收[batch, channel, height, width]，
    # 先保留batch维度，把每张图片展开为784个特征。
    flat_images = images.flatten(start_dim=1)

    model = TwoLayerClassifier(
        input_dim=28 * 28,
        hidden_dim=128,
        num_classes=10,
    )

    logits = model(images)
    loss = F.cross_entropy(logits, labels)

    print("flat_images.shape:", flat_images.shape)
    print("logits.shape:", logits.shape)
    print("loss:", loss.item())

if __name__ == "__main__":
    main()