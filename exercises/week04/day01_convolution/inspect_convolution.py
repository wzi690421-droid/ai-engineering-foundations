import torch
from torch import nn

images = torch.randn(8,3,32,32)

conv = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=1,
)

feature_maps = conv(images)

# ReLU把负特征值变成0，正数保持不变。
relu = nn.ReLU()
activated_feature_maps = relu(feature_maps)

pool = nn.MaxPool2d(
    kernel_size=2,
    stride=2,
)

# 从激活后的局部区域中保留最大响应。
pooled_feature_maps = pool(activated_feature_maps)

print("输入形状：", images.shape)
print("卷积参数形状：", conv.weight.shape)
print("卷积输出形状：", feature_maps.shape)
print("ReLU输出形状：", activated_feature_maps.shape)
print("池化输出形状：", pooled_feature_maps.shape)
print("ReLU前最小值：", feature_maps.min().item())
print("ReLU后最小值：", activated_feature_maps.min().item())

# 手动构造一张只有一个通道的4×4特征图。
# 四层中括号分别对应：[图片数量, 通道数量, 高度, 宽度]。
small_feature_map = torch.tensor(
    [
        [
            [
                [1.0, 5.0, 2.0, 0.0],
                [3.0, 4.0, 1.0, 2.0],
                [0.0, 2.0, 8.0, 1.0],
                [1.0, 3.0, 2.0, 6.0],
            ]
        ]
    ]
)

small_pooled = pool(small_feature_map)

print("池化前：")
print(small_feature_map[0, 0])

print("池化后：")
print(small_pooled[0, 0])
