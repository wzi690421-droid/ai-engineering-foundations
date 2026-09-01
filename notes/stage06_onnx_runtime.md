# 阶段 06：ONNX 导出与独立 Runtime

## 这一阶段解决了什么

目标是让冻结的 PyTorch 模型脱离 PyTorch 和 `model.py` 运行，并证明导出前后的数学结果一致。

```text
best_model.pt + SmallCNN Python结构
→ 导出 ONNX（计算图 + 权重 + 输入输出契约）
→ ONNX checker 检查
→ PyTorch / ONNX Runtime logits 对齐
→ 独立 NumPy 图片预处理
→ ONNX Runtime 图片分类
```

## `.pt`、ONNX 和 Runtime 的边界

- `best_model.pt`：当前只保存 PyTorch 参数；没有模型类代码就不知道这些参数如何连接。
- `.onnx`：保存计算图、参数和输入输出接口，是 Protobuf 二进制文件，不是可直接阅读的文本。
- `InferenceSession`：读取 ONNX、选择 CPU Execution Provider、优化并准备执行图。
- `session.run()`：真正把输入张量送入计算图并取得输出。

“ONNX 可以直接使用”表示部署端不再需要 PyTorch、`SmallCNN` 类和 `.pt`；不表示模型文件自己会执行。类似视频文件仍需要播放器加载。

## 当前 ONNX 图包含什么

SmallCNN 主图：

```text
Conv → Relu → MaxPool
→ Conv → Relu → MaxPool
→ Reshape
→ Gemm（Linear）
→ logits
```

图中没有图片解码、缩放、归一化和 Softmax，因为导出的是 `SmallCNN.forward()`，这些处理原本就在模型外面。

`Gemm` 可理解为：

```text
logits = x @ weight.T + bias
```

## 固定 batch 与动态 batch

固定模型输入：

```text
[1,3,32,32]
```

输入 8 张时会报 `Got 8, Expected 1`。使用：

```python
batch = torch.export.Dim("batch", min=1)
dynamic_shapes={"images": {0: batch}}
```

后，接口变为：

```text
['batch',3,32,32] → ['batch',10]
```

只有第 0 维可变；通道和高宽仍固定。

动态图多出：

```text
Shape(images) → 运行时读取 N
常量          → [2048]
Concat        → 构造 [N,2048]
Reshape       → [N,32,8,8] 变为 [N,2048]
```

`Concat` 在这里拼接的是“形状数字”，不是拼接图片。

## 跨后端正确性

必须把同一个输入同时交给 PyTorch 与 ONNX Runtime，并比较全部 logits：

```python
np.testing.assert_allclose(
    pytorch_logits,
    ort_logits,
    rtol=1e-5,
    atol=1e-5,
)
```

只比较 top-1 不够：两个后端即使数值严重不同，也可能碰巧选中同一最大下标。

浮点计算顺序和底层实现可能不同，因此不要求逐位一致。实测 batch `1、8、64` 全部通过，最大绝对误差约 `3.34×10⁻⁶`。

## 独立图片预处理

普通图片到模型输入：

```text
读取并转 RGB
→ resize 到 32×32
→ uint8 转 float32
→ 除以 255：0～255 变为 0～1
→ transpose(2,0,1)：HWC 变为 CHW
→ (x - mean) / std
→ 增加 batch 维
→ 连续 float32 数组 [N,3,32,32]
```

`mean.reshape(3,1,1)` 不会重新计算均值，只是把 `[3]` 改成可与 `[3,32,32]` 正确广播的形状，让 RGB 三个通道分别使用自己的统计量。

归一化后可以小于 0 或大于 1。忘记除以 255 通常不会报形状错误，但会把数值分布放大约 255 倍，模型输出不可信。

多张图片时，每张预处理结果为 `[1,3,32,32]`：

```text
concatenate(axis=0) → [N,3,32,32]，正确
stack(axis=0)       → [N,1,3,32,32]，多出一维
```

当前程序只调用一次 `session.run()`，因此 batch=3 是一次批量推理，不是循环推理三次。

## logits、Softmax 与 top-3

Runtime 输出 `[N,10]` logits。对每张图片分别执行稳定 Softmax：

```python
shifted = logits - logits.max(axis=1, keepdims=True)
probabilities = exp(shifted) / exp(shifted).sum(axis=1, keepdims=True)
```

`axis=1` 表示在每张图片自己的 10 个类别之间归一化；`axis=0` 会错误地让不同图片互相影响。

官方测试集索引 0 的真实标签是 `cat`，独立 ONNX 推理结果：

```text
cat：  76.37%
dog：  15.14%
ship：  3.23%
```

这证明一张图片的端到端链路正确，但不能据此重新证明整个测试集准确率为 70.86%。

## 常见误区

- 导出时的示例输入帮助追踪图和确定接口，通常不作为样本保存在 ONNX 中。
- `onnx.load()` 用于读取和检查图；`InferenceSession()` 用于真正准备推理。
- ONNX 本身可以运行不等于预处理可以随便改；错误 mean/std 仍能运行，但结果不可信。
- `Shape + Concat + Reshape` 是为了动态 Flatten，不是普通图片 batch 拼接。

## 已完成证据与薄弱点

- ONNX checker 通过，固定与动态 batch 行为验证通过。
- PyTorch/ORT 全 logits 对齐，独立 Runtime 不导入 PyTorch。
- NumPy 预处理、真实图片、Softmax、top-3 和动态 batch 运行通过。
- 阶段评分 `3.55/5`。
- 仍需巩固：动态图辅助节点、广播语义，以及把 Python 预处理严格迁移到后续 C++ Runtime。

## 复盘自测

1. `.pt`、`.onnx`、`InferenceSession` 分别负责什么？
2. 为什么动态 Flatten 需要 Shape 和 Concat？
3. 为什么跨后端验证要比较全部 logits？
4. `mean.reshape(3,1,1)` 改变了数值还是只改变形状？
5. `concatenate` 与 `stack` 在 batch 拼接时有什么区别？
