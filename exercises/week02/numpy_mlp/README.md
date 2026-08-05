# Week 02：NumPy 两层神经网络

## 项目结构

```text
numpy_mlp/
├── model.py       # 线性层、激活函数、损失、前向与反向传播
├── test_model.py  # 独立测试入口
├── train.py       # 训练入口，Day 6 增加
└── README.md      # 学习目标、运行方式和实验结论
```

`model.py` 只放模型数学逻辑，`test_model.py` 只放测试，后续 `train.py` 只负责数据与训练流程。

## Day 1：NumPy 形状与线性层

## 今日目标

- 理解批量输入、权重、偏置和输出的形状关系。
- 不使用循环，实现线性层前向传播 `Y = XW + b`。
- 主动检查非法形状，并用测试证明广播行为正确。

## 形状约定

```text
X: (N, D)  N 个样本，每个样本 D 个特征
W: (D, H)  从 D 维输入映射到 H 维输出
b: (H,)    每个输出维度一个偏置
Y: (N, H)  N 个样本的 H 维输出
```

矩阵乘法只允许内部维度匹配：`X` 的第二维必须等于 `W` 的第一维。`b` 通过广播加到每一行。

## 第一部分：先手算

给定：

```python
X = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
])

W = np.array([
    [ 1.0, 0.0, -1.0],
    [ 0.0, 2.0,  1.0],
])

b = np.array([1.0, -1.0, 0.0])
```

在写代码前记录 `X.shape`、`W.shape`、`b.shape`、`Y.shape`，并手算 `Y`。

## 第二部分：从空文件实现

在 `model.py` 中实现：

```python
def linear_forward(x, w, b):
    ...
```

要求：

1. 使用 NumPy；
2. 不使用 `for` 循环；
3. 不修改输入数组；
4. `x` 和 `w` 必须是二维数组；
5. `b` 必须是一维数组；
6. 矩阵乘法内部维度必须匹配；
7. `b` 的长度必须等于输出维度；
8. 非法输入抛出带有明确说明的 `ValueError`。

## 第三部分：测试

至少覆盖：

- 上面的固定数值样例；
- 只有一个样本时输出仍保持二维；
- `x` 不是二维；
- `w` 不是二维；
- 矩阵内部维度不匹配；
- 偏置长度不匹配；
- 函数调用前后输入数组保持不变。

浮点数组使用：

```python
np.testing.assert_allclose(actual, expected)
```

## 运行

```bash
python3 test_model.py
```

全部通过时输出：

```text
Week 2 Day 1-3 model functions passed
```

## 完成后解释

1. 为什么 `(N, D) @ (D, H)` 得到 `(N, H)`？
2. 为什么 `(H,)` 可以加到 `(N, H)`？
3. `(H, 1)` 能否作为这里的偏置？实际会发生什么？
4. 为什么测试浮点结果更适合使用 `assert_allclose` 而不是 `==`？

## Day 2：ReLU、Softmax 与交叉熵

已实现：

- `relu`：引入非线性，输入输出形状不变。
- `softmax`：逐行生成概率，减去行最大值避免指数溢出。
- `cross_entropy`：从概率计算平均分类损失。
- `cross_entropy_from_logits`：通过 log-sum-exp 直接计算稳定且准确的损失。

关键数据流：

```text
logits (N, C)
  → softmax probabilities (N, C)
  → 选取 N 个正确类别分数
  → N 个样本损失
  → 平均标量 loss
```

## Day 3：两层网络前向传播

已实现 `two_layer_forward`：

```text
x (N, D)
  → Linear 1 → z1 (N, H)
  → ReLU     → h1 (N, H)
  → Linear 2 → logits (N, C)
  → Softmax  → probabilities (N, C)
```

前向传播保存 `x、z1、h1、logits、probabilities` 到 `cache`，供后续反向传播使用。固定输入的中间形状、概率和逐行归一结果已通过验收。
