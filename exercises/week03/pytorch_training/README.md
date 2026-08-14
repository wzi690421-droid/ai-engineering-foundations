# Week 03 Day 01：PyTorch 自动求导对照

## 今天只解决一个问题

把 Week 02 手写的两层网络换成 PyTorch 表达，并验证 PyTorch 自动得到的四组梯度与 NumPy 手算结果一致。

流程：

```text
固定输入和参数
→ PyTorch 前向传播
→ CrossEntropyLoss
→ loss.backward()
→ 读取 w1、b1、w2、b2 的 .grad
→ 与 NumPy 梯度比较
```

## 运行环境

仓库根目录的 `.venv/` 是本项目独立 Python 环境，不提交 Git。

从仓库根目录运行：

```bash
.venv/bin/python exercises/week03/pytorch_training/autograd_compare.py
```

## Day 01 通过标准

1. 能解释 Tensor、`dtype`、`requires_grad` 和 `.grad`。
2. 能独立写出两层网络的 PyTorch 前向传播。
3. 能解释为什么交叉熵直接接收 `logits`，不先手动调用 Softmax。
4. 能说明 `loss.backward()` 做了什么，以及为什么梯度会累积。
5. PyTorch 与 NumPy 的 loss 和四组梯度在约定误差内一致。

固定数据和测试对照可以由 Codex 提供；前向传播、损失和反向调用由学习者完成。
