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
Week 2 Day 1-6 NumPy MLP passed
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

## Day 4：反向传播

已实现 `two_layer_backward`：

```text
loss
  → dlogits
  → dw2、db2、dh1
  → ReLU mask → dz1
  → dw1、db1
```

参数梯度形状与原参数一致；固定样例中被 ReLU 关闭的第三个隐藏神经元，对应的输入权重、偏置和输出权重梯度均为零。

## Day 5：梯度检查

`gradient_check.py` 使用中心有限差分近似数值梯度：

```text
numerical_gradient = (loss(parameter + epsilon)
                    - loss(parameter - epsilon)) / (2 * epsilon)
```

它逐个检查 `w1、b1、w2、b2`，与反向传播的解析梯度比较，相对误差均约为 `1e-11`。

`gradient_bug_lab.py` 是独立故障练习，故意遗漏 ReLU 反向掩码。梯度检查显示 `dw2/db2` 通过而 `dw1/db1` 失败，因此可将错误定位到第二层参数梯度之后、第一层参数梯度之前。

运行：

```bash
python3 gradient_check.py
python3 gradient_bug_lab.py
```

## Day 6：训练循环与学习率实验

`train.py` 已完成：

- 生成可复现的 XOR 合成二分类数据。
- 使用 He 初始化创建 `2 → 8 → 2` 网络。
- 分离不更新参数的 `evaluate` 与执行梯度下降的 `train_step`。
- 使用独立随机种子生成训练数据和测试数据。
- 控制其他条件不变，对比三组学习率并保存原始 CSV。

500 步后的实验结论：

| 学习率 | 训练损失 | 训练准确率 | 测试损失 | 测试准确率 | 结论 |
|---:|---:|---:|---:|---:|---|
| 0.001 | 0.765505 | 0.500 | 0.767779 | 0.500 | 在学习，但收敛过慢 |
| 0.1 | 0.009322 | 1.000 | 0.012612 | 1.000 | 快速、稳定收敛 |
| 10.0 | 2.471274 | 0.500 | 2.471274 | 0.500 | 步长过大，无法有效优化 |

运行：

```bash
python3 train.py
```

原始记录保存为 `learning_rate_results.csv`。测试集只用于 `evaluate`，不参与梯度更新。
