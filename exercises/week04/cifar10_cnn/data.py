from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)


class TransformedSubset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        indices: list[int],
        transform,
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        original_index = self.indices[position]
        image, label = self.dataset[original_index]
        image = self.transform(image)

        return image, label


def compute_channel_statistics(
    dataset: Dataset,
    indices: list[int],
    batch_size: int = 512,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    # 统计阶段只转换为[0, 1]范围的Tensor，不能提前标准化。
    statistics_dataset = TransformedSubset(
        dataset=dataset,
        indices=indices,
        transform=ToTensor(),
    )
    statistics_loader = DataLoader(
        statistics_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # 分别累计三个颜色通道的像素和与像素平方和。
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_squared_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0

    for images, _ in statistics_loader:
        images = images.to(torch.float64)
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += images.square().sum(dim=(0, 2, 3))

        # 每个颜色通道在这个批次中都有N×H×W个像素。
        pixel_count += (
            images.shape[0]
            * images.shape[2]
            * images.shape[3]
        )

    mean = channel_sum / pixel_count
    variance = channel_squared_sum / pixel_count - mean.square()
    std = torch.sqrt(variance.clamp_min(0.0))

    return tuple(mean.tolist()), tuple(std.tolist())


def create_evaluation_transform(
    mean: tuple[float, ...],
    std: tuple[float, ...],
):
    return Compose(
        [
            ToTensor(),
            Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )


def create_training_transform(
    mean: tuple[float, ...],
    std: tuple[float, ...],
    random_crop_padding: int = 4,
    horizontal_flip_probability: float = 0.5,
):
    return Compose(
        [
            # 先在四周补边，再随机裁回32×32，相当于轻微随机平移。
            RandomCrop(32, padding=random_crop_padding),
            # 以给定概率左右翻转；CIFAR-10类别不会因此改变。
            RandomHorizontalFlip(p=horizontal_flip_probability),
            ToTensor(),
            Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )


def create_train_validation_loaders(
    data_dir: Path,
    batch_size: int = 64,
    seed: int = 42,
    validation_size: int = 5_000,
    use_data_augmentation: bool = True,
    random_crop_padding: int = 4,
    horizontal_flip_probability: float = 0.5,
) -> tuple[
    DataLoader,
    DataLoader,
    tuple[float, ...],
    tuple[float, ...],
]:
    # 官方50000张训练图片。
    full_train_dataset = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=None,
    )

    total_size = len(full_train_dataset)

    if not 0 < validation_size < total_size:
        raise ValueError(
            f"validation_size必须在1到{total_size - 1}之间，"
            f"当前值为{validation_size}"
        )

    if random_crop_padding < 0:
        raise ValueError("random_crop_padding不能是负数")

    if not 0.0 <= horizontal_flip_probability <= 1.0:
        raise ValueError(
            "horizontal_flip_probability必须在0到1之间"
        )

    train_size = total_size - validation_size

    # 只控制训练集/验证集的成员划分。
    # 相同seed会得到完全相同的训练集和验证集成员。
    split_generator = torch.Generator().manual_seed(seed)

    train_index_subset, validation_index_subset = random_split(
        full_train_dataset,
        lengths=[train_size, validation_size],
        generator=split_generator,
    )

    # 均值和标准差只由训练下标对应的图片估计。
    mean, std = compute_channel_statistics(
        dataset=full_train_dataset,
        indices=train_index_subset.indices,
    )

    if use_data_augmentation:
        training_transform = create_training_transform(
            mean=mean,
            std=std,
            random_crop_padding=random_crop_padding,
            horizontal_flip_probability=horizontal_flip_probability,
        )
    else:
        training_transform = create_evaluation_transform(mean, std)

    # 两个包装对象共享原始图片，但训练变换随机、验证变换固定。
    train_dataset = TransformedSubset(
        dataset=full_train_dataset,
        indices=train_index_subset.indices,
        transform=training_transform,
    )
    validation_dataset = TransformedSubset(
        dataset=full_train_dataset,
        indices=validation_index_subset.indices,
        transform=create_evaluation_transform(mean, std),
    )

    # 单独控制训练集每个epoch的打乱顺序。
    # 每个epoch顺序不同，但整个顺序序列可以复现。
    shuffle_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
    )

    # 验证只用于评估，因此不需要打乱。
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, validation_loader, mean, std


def create_test_loader(
    data_dir: Path,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    batch_size: int = 64,
) -> DataLoader:
    # 测试集只在最终评估时按需创建。
    test_dataset = CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        # 测试集只能使用训练集估计出的统计量。
        transform=create_evaluation_transform(mean, std),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return test_loader
