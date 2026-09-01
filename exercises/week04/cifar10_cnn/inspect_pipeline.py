from pathlib import Path

import torch
import torch.nn.functional as F

from data import create_train_validation_loaders
from model import SmallCNN


def main():
    # 固定模型初始化，保证重新运行时结果可以复现。
    torch.manual_seed(42)

    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data"

    train_loader, _, _, _ = create_train_validation_loaders(
        data_dir=data_dir,
        batch_size=64,
        seed=42,
    )

    # 取出一批真实CIFAR-10图片和标签。
    images, labels = next(iter(train_loader))

    # 创建尚未训练的CNN。
    model = SmallCNN()

    # 当前只检查前向传播，不训练，因此切换到评估模式。
    model.eval()

    # 不记录梯度，减少不必要的内存占用。
    with torch.no_grad():
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

    # 在10个logits中选择分数最大的类别。
    predictions = logits.argmax(dim=1)

    # 比较预测和真实标签，计算当前batch的准确率。
    accuracy = (
        (predictions == labels)
        .float()
        .mean()
        .item()
    )

    print("图片形状：", images.shape)
    print("标签形状：", labels.shape)
    print("logits形状：", logits.shape)

    print("前5个真实标签：", labels[:5])
    print("前5个预测标签：", predictions[:5])

    print("初始损失：", loss.item())
    print("初始准确率：", accuracy)


if __name__ == "__main__":
    main()
