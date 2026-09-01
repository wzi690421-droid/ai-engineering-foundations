from pathlib import Path

from data import create_test_loader, create_train_validation_loaders


def main():
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data"

    # 数据下载、预处理和固定划分都由data.py统一负责。
    train_loader, validation_loader, mean, std = (
        create_train_validation_loaders(
            data_dir=data_dir,
            batch_size=64,
            seed=42,
        )
    )
    test_loader = create_test_loader(
        data_dir=data_dir,
        mean=mean,
        std=std,
        batch_size=64,
    )

    # 只取训练集的第一批，检查数据管线是否工作正常。
    images, labels = next(iter(train_loader))

    # test_loader直接持有官方CIFAR10数据集，可以取得类别名称。
    class_names = test_loader.dataset.classes

    print("训练样本数量：", len(train_loader.dataset))
    print("验证样本数量：", len(validation_loader.dataset))
    print("测试样本数量：", len(test_loader.dataset))

    print("训练批次数量：", len(train_loader))
    print("验证批次数量：", len(validation_loader))
    print("测试批次数量：", len(test_loader))

    print("类别名称：", class_names)
    print("图片batch形状：", images.shape)
    print("标签batch形状：", labels.shape)
    print("图片数值范围：", images.min().item(), images.max().item())
    print("训练集通道均值：", mean)
    print("训练集通道标准差：", std)
    print("训练集变换：", train_loader.dataset.transform)
    print("验证集变换：", validation_loader.dataset.transform)

    print("前8个标签：", labels[:8])
    print(
        "前8个类别：",
        [class_names[label.item()] for label in labels[:8]],
    )


if __name__ == "__main__":
    main()
