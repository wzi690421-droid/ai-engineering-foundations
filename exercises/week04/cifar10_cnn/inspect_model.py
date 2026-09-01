import torch

from model import SmallCNN


# 创建模型对象。
# 这一步会初始化两个卷积层和一个线性分类层的参数。
model = SmallCNN()

# 模拟一次输入8张32×32的RGB图片。
images = torch.randn(8, 3, 32, 32)

# 不要直接调用model.forward(images)。
# model(images)会通过PyTorch的框架流程调用forward()。
logits = model(images)

# 查看完整模型结构。
print(model)

print("输入形状：", images.shape)
print("输出logits形状：", logits.shape)

# model.parameters()依次提供模型中的所有可训练参数。
# parameter.numel()返回一个参数张量中包含多少个数字。
parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("可训练参数总数：", parameter_count)
