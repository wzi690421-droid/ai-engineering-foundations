# 阶段 02：NumPy 两层神经网络

## 这一阶段解决了什么

目标是看清神经网络内部的真实计算，而不是直接调用框架。最终用 NumPy 完成两层分类网络、反向传播、梯度检查和训练循环。

```text
输入 x
→ 线性层 z1 = x @ w1 + b1
→ ReLU 得到 h1
→ 线性层 logits = h1 @ w2 + b2
→ Softmax / 交叉熵
→ 反向传播得到四组参数梯度
→ 梯度下降更新参数
```

## 形状是第一检查项

```text
x：      [N,D]  N 个样本，每个样本 D 个特征
w1：     [D,H]
b1：     [H]
z1/h1：  [N,H]
w2：     [H,C]
b2：     [C]
logits： [N,C]
```

矩阵乘法要求内部维度一致：`[N,D] @ [D,H] → [N,H]`。偏置 `[H]` 通过广播加到每一行，而不是与权重相乘。

## 激活、概率与损失

ReLU：

```text
z > 0 → 输出 z
z ≤ 0 → 输出 0
```

如果没有非线性，多层线性变换仍可合并成一层线性变换，表达能力不会真正增加。ReLU 反向传播必须依据前向的 `z1` 判断开关，不能依据上游梯度 `dh1`。

Softmax 对每个样本的类别维计算：

```python
shifted = logits - logits.max(axis=1, keepdims=True)
probabilities = exp(shifted) / exp(shifted).sum(axis=1, keepdims=True)
```

减去最大值不会改变概率，因为分子分母会同时乘上同一个因子；它只用于避免 `exp` 溢出。

平均交叉熵：

\[
L=\frac{1}{N}\sum_i-\log p_{i,y_i}
\]

实际实现更适合直接从 logits 使用 log-sum-exp，避免先得到极小概率再取对数造成数值问题。

## 反向传播主链路

Softmax 与交叉熵合并求导后：

```python
dlogits = probabilities.copy()
dlogits[np.arange(N), labels] -= 1
dlogits /= N
```

正确类别位置减一，使该位置梯度通常为负；梯度下降 `parameter -= learning_rate * gradient` 会提高正确类别对应分数。除以 `N` 是因为损失定义为批次平均值。

第二层：

```text
dw2 = h1.T @ dlogits
db2 = dlogits.sum(axis=0)
dh1 = dlogits @ w2.T
```

第一层：

```text
dz1 = dh1 * (z1 > 0)
dw1 = x.T @ dz1
db1 = dz1.sum(axis=0)
```

前向传播保存 `x、z1、h1、logits、probabilities` 到 cache，是因为反向传播需要使用同一轮前向的中间值。

## 梯度检查

中心有限差分：

\[
g_{num}=\frac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon}
\]

每次只扰动一个参数元素，计算后必须恢复原值。它很慢，不用于训练；它用于验证手写解析梯度是否正确。

故障实验中故意漏掉 ReLU 掩码：`dw2/db2` 通过，而 `dw1/db1` 失败。因此错误范围可缩小到“第二层梯度之后、第一层梯度之前”。调试梯度不是盲看整个网络，而是利用链路分段定位。

## 训练循环与实验判断

一轮更新必须使用同一版本参数：

```text
前向 → loss → 全部梯度 → 一次性更新全部参数
```

不能算出一个梯度就立刻更新对应参数，否则后续梯度会混用新旧参数。

- 学习率太小：方向可能正确，但有限步数内几乎不收敛。
- 学习率合适：损失稳定下降并较快收敛。
- 学习率太大：反复越过较优区域，损失震荡或失败。
- 测试数据只用于评估，不能参与梯度更新。

XOR 实验中，`0.001` 收敛过慢，`0.1` 快速达到 100%，`10.0` 无法有效学习。结论来自控制其他条件相同的对照，而不是只看一次结果。

## 常见误区

- loss 是标量；`logits/probabilities` 是 `[N,C]`，每个样本的损失是 `[N]`。
- `cross_entropy_from_logits()` 虽然返回 loss，反向传播仍可使用化简后的 `probabilities - one_hot`；不是“前面的 loss 没用”。
- `db` 对 batch 维求和；如果损失取平均，整条梯度最终都应包含 `/N`。
- Leaky ReLU 的负区间导数是 `negative_slope`，判断依据仍是前向输入 `z1`。

## 已完成证据与薄弱点

- 前向、反向、数值稳定损失和四组参数梯度测试通过。
- 梯度检查相对误差约 `1e-11`。
- 完成学习率对照和独立测试，阶段评分 `3/5`。
- 仍需巩固：交叉熵梯度推导、广播细节、按 batch 求平均，以及不用提示迁移到新激活函数。

## 复盘自测

1. 为什么 `b1.shape == [H]` 可以加到 `[N,H]`？
2. 为什么 Softmax 必须沿类别维 `axis=1`？
3. `dlogits` 为什么在正确类别位置减一并除以 `N`？
4. 为什么更新参数必须等所有梯度计算完成？
5. 如果只有 `dw1/db1` 梯度检查失败，应优先检查哪一段？
