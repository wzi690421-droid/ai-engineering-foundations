# Week 02 Day 07 周测

## 当前进度

| 环节 | 状态 | 独立完成情况与提示记录 |
|---|---|---|
| 前向传播与损失 | 通过 | 独立完成线性层、ReLU、稳定 Softmax 和两层前向传播；交叉熵经过数值稳定性和数组形状提示后修正 |
| 反向传播 | 通过 | 独立写出完整反向链路；经过“按批次求平均”的一次提示后修正 |
| Leaky ReLU 需求变化 | 通过 | 最初使用普通 `if` 判断 NumPy 数组，并一度根据 `dh1` 判断导数；经过两轮错误反馈后改为依据 `z1` 计算导数 |
| 独立测试 | 通过 | `test_mlp_core.py` 由 Codex 编写，覆盖固定样例、极端值、缓存，以及 ReLU 和 Leaky ReLU 的四组参数数值梯度 |
| 完整训练闭环 | 通过 | `train_step` 及独立更新测试由 Codex 编写；学习者确认已理解，因此作为流程理解项，不计入独立编码得分 |
| 口头答辩与复盘 | 通过 | 能说明前向形状、Leaky ReLU 局部导数和参数更新一致性；交叉熵梯度推导、批次平均及数值梯度隔离经过讲解后理解 |

当前结论：全部编码测试和口头答辩通过，Week 02 Day 07 完成，进入第 3 周。

## 最终结果

综合评分：**3/5，通过。**

评分依据：

- 能从空文件独立完成线性层、ReLU、稳定 Softmax、两层前向传播和完整反向传播主体。
- 能正确说明主要张量形状、激活函数的局部梯度，以及一次参数更新必须使用同一版本参数。
- 交叉熵的稳定实现、批次平均梯度和 Leaky ReLU 反向传播经过提示后修正，尚未达到完全独立迁移。
- 数值梯度的中心差分原理已经理解，但对“每次只改变一个元素并恢复原值”和按失败梯度缩小排查范围仍需巩固。
- `train_step` 由 Codex 代写并通过测试，只作为已理解的流程胶水，不计入独立编码成绩。

进入第 3 周后，不重做整套 NumPy MLP。通过 PyTorch 的自动求导、训练循环和独立验证流程，针对性复测上述薄弱点。

## 第 1 段：闭卷复现前向传播与损失

建议限时：50 分钟。

规则：

1. 不查看 `numpy_mlp/model.py`、`train.py` 或以前的答案。
2. 可以使用 VS Code 普通语法补全，但不使用 AI 生成实现。
3. 只编辑 `mlp_core.py`，不要创建测试代码。
4. 暂时不写反向传播；完成后由 Codex 添加独立测试并验收。

需要独立实现：

1. `linear_forward(x, w, b)`：完成线性层前向传播。
2. `relu(x)`：完成 ReLU。
3. `stable_softmax(logits)`：按行计算数值稳定的 Softmax。
4. `cross_entropy_from_logits(logits, labels)`：返回一个批次的平均交叉熵损失。
5. `two_layer_forward(x, w1, b1, w2, b2)`：组合两层网络，返回 `probabilities, cache`。

约定：

- `x.shape == (batch_size, input_size)`
- `w1.shape == (input_size, hidden_size)`
- `b1.shape == (hidden_size,)`
- `w2.shape == (hidden_size, class_count)`
- `b2.shape == (class_count,)`
- `labels.shape == (batch_size,)`，每个元素是正确类别编号
- `cache` 至少保存反向传播需要的 `x、z1、h1、logits、probabilities`

本阶段不要求额外形状检查，不要求写 `main()`，也不要求打印结果。

## 第 2 段：闭卷复现反向传播

建议限时：45 分钟。

继续只编辑 `mlp_core.py`，实现：

```python
two_layer_backward(labels, cache, w2)
```

要求：

1. 使用第 1 段前向传播产生的 `cache`。
2. 所有梯度按 `batch_size` 求平均。
3. 返回包含 `dw1、db1、dw2、db2` 的字典。
4. 返回形状必须分别匹配 `w1、b1、w2、b2`。
5. 不查看旧实现，不修改已经通过的前向函数。

## 第 2 段需求变化：支持 Leaky ReLU

建议限时：30 分钟。

在保持旧调用方式可用的前提下，修改当前两层网络：

1. `two_layer_forward` 增加可选参数 `negative_slope=0.0`。
2. 隐藏层激活改为：当 `z1 > 0` 时输出 `z1`，否则输出 `negative_slope * z1`。
3. 把 `negative_slope` 存入 `cache`。
4. `two_layer_backward` 根据 `cache` 中的斜率计算对应激活梯度。
5. 当调用者不传参数时，行为必须与原来的 ReLU 完全一致。

本题不要求修改独立的 `relu(x)` 函数。完成后会同时运行旧测试、Leaky ReLU固定样例和数值梯度检查。
