from torch import nn


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 第一组：RGB图片 → 16种初级特征 → 空间尺寸减半。
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二组：16种初级特征 → 32种更复杂特征 → 再次减半。
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 将32张8×8特征图展开，然后输出10个分类分数。
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(32 * 8 * 8, 10)

    def forward(self, images):
        # images: [N, 3, 32, 32]

        # 第一组卷积：提取16种初级特征。
        x = self.conv1(images)       # [N, 16, 32, 32]
        x = self.relu1(x)            # [N, 16, 32, 32]
        x = self.pool1(x)            # [N, 16, 16, 16]

        # 第二组卷积：组合成32种更复杂的特征。
        x = self.conv2(x)            # [N, 32, 16, 16]
        x = self.relu2(x)            # [N, 32, 16, 16]
        x = self.pool2(x)            # [N, 32, 8, 8]

        # Linear不能直接接收四维图片，因此保留批次维，
        # 将每张图片的32×8×8个特征展开成长度2048的向量。
        x = self.flatten(x)          # [N, 2048]

        # 为每张图片输出10个类别分数。
        logits = self.classifier(x)  # [N, 10]

        return logits

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # MLP不能直接处理四维图片，首先展开每张图片。
        self.flatten = nn.Flatten(start_dim=1)

        # 3072来自3×32×32。
        # 隐藏层取8，是为了让MLP参数量接近SmallCNN。
        self.hidden = nn.Linear(3 * 32 * 32, 8)
        self.relu = nn.ReLU()
        self.output = nn.Linear(8, 10)

    def forward(self, images):
        x = self.flatten(images)  # [N,3,32,32] → [N,3072]
        x = self.hidden(x)        # [N,3072] → [N,8]
        x = self.relu(x)          # 形状不变
        logits = self.output(x)   # [N,8] → [N,10]

        return logits

class SimpleResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        # 输入和输出通道都等于channels，
        # 这样卷积分支的输出才能与原输入逐元素相加。
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        # 保存进入残差块时的原始特征。
        identity = x

        # 卷积分支计算F(x)。
        correction = self.conv1(x)
        correction = self.relu(correction)
        correction = self.conv2(correction)

        # 原始特征加上卷积分支学到的修正量。
        output = identity + correction

        return output

class DownsamplingResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        # 卷积分支：提取特征，并把空间尺寸减半。
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        # 捷径分支：不负责复杂特征提取，只负责对齐形状。
        self.shortcut = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=2,
        )

    def forward(self, x):
        # 卷积分支F(x)。
        correction = self.conv1(x)
        correction = self.relu(correction)
        correction = self.conv2(correction)

        # 捷径分支也把x转换成相同形状。
        identity = self.shortcut(x)

        # 两边都是[N,out_channels,H/2,W/2]，现在可以相加。
        output = identity + correction

        return output

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()

        # 卷积分支的第一层。
        # stride=2时会把高度和宽度减半。
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # 第二层不再缩小尺寸，继续提取特征。
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果通道数或空间尺寸改变，捷径也必须进行投影。
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # 形状已经一致时，什么都不做，直接返回输入。
            self.shortcut = nn.Identity()

    def forward(self, x):
        # 捷径分支：直接通过，或者进行1×1投影。
        identity = self.shortcut(x)

        # 卷积分支F(x)。
        correction = self.conv1(x)
        correction = self.bn1(correction)
        correction = self.relu(correction)

        correction = self.conv2(correction)
        correction = self.bn2(correction)

        # 两条路线合并。
        output = identity + correction
        output = self.relu(output)

        return output

class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 将RGB图片转换成16通道的初级特征。
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # 保持形状：[N,16,32,32]。
        self.layer1 = ResidualBlock(
            in_channels=16,
            out_channels=16,
            stride=1,
        )

        # 通道增加，空间尺寸减半：
        # [N,16,32,32] → [N,32,16,16]。
        self.layer2 = ResidualBlock(
            in_channels=16,
            out_channels=32,
            stride=2,
        )

        # 再次增加通道并减半：
        # [N,32,16,16] → [N,64,8,8]。
        self.layer3 = ResidualBlock(
            in_channels=32,
            out_channels=64,
            stride=2,
        )

        # 将每张8×8特征图取平均，
        # [N,64,8,8] → [N,64,1,1]。
        self.global_pool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )

        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(64, 10)

    def forward(self, images):
        x = self.stem(images)       # [N,16,32,32]
        x = self.layer1(x)          # [N,16,32,32]
        x = self.layer2(x)          # [N,32,16,16]
        x = self.layer3(x)          # [N,64,8,8]

        x = self.global_pool(x)     # [N,64,1,1]
        x = self.flatten(x)         # [N,64]
        logits = self.classifier(x) # [N,10]

        return logits