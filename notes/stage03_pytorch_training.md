# 阶段 03：PyTorch 训练工程

## 这一阶段解决了什么

目标是把阶段 02 的数学映射到 PyTorch，并建立规范的训练、验证、保存和对照实验流程。数据集使用 Fashion-MNIST。

```text
Dataset
→ DataLoader 分批
→ nn.Module 前向传播
→ CrossEntropyLoss
→ autograd 计算梯度
→ optimizer 更新参数
→ 独立验证
→ 保存 latest / best
```

## 自动求导与参数

- `Tensor` 同时包含数值、形状、类型和设备信息。
- `nn.Parameter` 默认需要梯度；普通输入和标签默认不需要。
- 前向运算会建立计算图，`loss.backward()` 沿图使用链式法则，把结果累积到参数的 `.grad`。
- 梯度默认累积，因此每个 batch 更新前要 `optimizer.zero_grad()`。
- `optimizer.step()` 根据已有 `.grad` 修改参数，本身不负责计算梯度。

标准训练步骤：

```python
optimizer.zero_grad()
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

`CrossEntropyLoss` 直接接收 logits，内部已经包含稳定的 LogSoftmax 与 NLLLoss；训练时不要先手动 Softmax。

## 模型结构与形状

`TwoLayerClassifier` 的基本关系：

```text
[N,1,28,28]
→ Flatten
→ [N,784]
→ Linear + ReLU
→ Linear
→ [N,10] logits
```

`model(images)` 会自动调用 `forward()`。`nn.Module` 负责登记子层和参数，所以 `model.parameters()`、`state_dict()` 和设备迁移才能统一工作。

## DataLoader、batch 与 epoch

- batch size 是一次参数更新使用的样本数。
- 一个 epoch 表示完整遍历一次训练集，不是只训练一个 batch。
- 50,000 个样本、batch size 64 时，每个 epoch 有 `ceil(50000/64)=782` 个训练步骤，最后一批可以少于 64。
- `shuffle=True` 只是每个 epoch 重排同一训练集，不会生成新的样本，也不会把上一轮样本永久删除。
- 固定划分种子可保证训练/验证成员一致；训练 DataLoader 可以每轮重排顺序。

## 训练模式与评估模式

```python
model.train()
```

用于训练；会启用训练态 Dropout 和 BatchNorm 行为。

```python
model.eval()
with torch.inference_mode():
```

用于评估；`eval()` 修改层行为，`inference_mode()` 禁止记录梯度，减少时间和内存。两者作用不同，不能互相替代。

训练指标通常在参数更新过程中累计；验证指标是在 epoch 结束后用最终参数重新计算，因此二者不是完全同一时刻的模型状态。

## checkpoint 与模型文件

```python
torch.save(model.state_dict(), "best_model.pt")
```

只保存模型参数；加载时仍需要相同的模型 Python 结构。

用于续训的 checkpoint 至少保存：

```text
模型状态
优化器状态
已经完成的 epoch
当前/最佳验证指标
```

优化器状态很重要，因为动量等优化器可能记住历史更新信息。`latest_checkpoint` 用于接着最后进度训练；`best_model` 用于最终评估或推理，两者职责不同。

## 公平实验

比较学习率时，每组实验都应：

- 使用相同训练/验证划分；
- 使用相同模型初始参数；
- 使用相同数据顺序和 epoch 数；
- 只改变学习率。

实验中 `lr=0.1` 在 5 个 epoch 达到 85.56% 验证准确率，高于 `lr=0.01` 的 82.12%。这只说明当前设置下前者收敛更快，不代表所有任务都应使用 0.1。

## Python 导入与环境

- `import torch` 只把顶层模块绑定为 `torch`；仍可以写 `torch.nn`、`torch.utils.data.DataLoader`。
- `from torch import nn` 把 `nn` 直接绑定到当前文件，减少重复前缀。
- `from torch.utils.data import DataLoader` 只直接导入需要的名称。
- 项目使用根目录 `.venv`，VS Code 解释器与终端环境必须指向同一个 Python，避免“终端能运行、编辑器报缺包”。

## 常见误区

- `requires_grad=True` 不是“默认所有数据都求梯度”；通常只有可训练参数需要。
- 重新启动程序若只重建模型，会从头开始；想续训必须加载 checkpoint。
- 一个 epoch 重复使用同一训练集是正常优化过程，不是数据使用错误。
- 验证集用于选超参数和最佳模型；不能送进训练循环参与更新。

## 已完成证据与薄弱点

- PyTorch 与 NumPy 的 loss 和四组梯度对齐。
- Fashion-MNIST 完整训练、验证、latest/best 保存与加载通过。
- 学习率公平对照生成 CSV 和曲线，阶段评分 `3/5`。
- 仍需巩固：独立设计 checkpoint、随机状态的严格恢复，以及减少照着现有函数拼训练流程。

## 复盘自测

1. `loss.backward()`、`optimizer.step()` 和 `zero_grad()` 分别做什么？
2. 为什么 CrossEntropyLoss 前不应手动 Softmax？
3. `model.eval()` 与 `inference_mode()` 有什么区别？
4. 为什么一个 epoch 会有 782 次参数更新？
5. `best_model` 和 `latest_checkpoint` 分别服务什么场景？
