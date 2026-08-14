import torch
import torch.nn.functional as F


def main():
    # x 是三个样本，每个样本有两个输入特征。
    # 我们不需要优化输入数据，所以 x 不开启梯度记录。
    x = torch.tensor(
        [[0.2, -0.4], [1.0, 0.5], [-0.3, 0.8]],
        dtype=torch.float64,
    )

    # labels 保存每个样本的正确类别编号，必须使用整数类型。
    labels = torch.tensor([0, 1, 0], dtype=torch.long)

    # 下面四个 Tensor 是需要训练的参数，因此开启 requires_grad。
    # float64 便于稍后和 NumPy 数值梯度进行高精度比较。
    w1 = torch.tensor(
        [[0.6, -0.2, 0.4], [-0.5, 0.7, 0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    b1 = torch.tensor(
        [0.2, 0.1, -0.4],
        dtype=torch.float64,
        requires_grad=True,
    )
    w2 = torch.tensor(
        [[0.2, -0.4], [0.5, 0.3], [-0.2, 0.6]],
        dtype=torch.float64,
        requires_grad=True,
    )
    b2 = torch.tensor(
        [0.1, -0.2],
        dtype=torch.float64,
        requires_grad=True,
    )

    print("x.shape:", x.shape)
    print("labels.shape:", labels.shape)
    print("w1.shape:", w1.shape)
    print("b1.shape:", b1.shape)
    print("w2.shape:", w2.shape)
    print("b2.shape:", b2.shape)

    z1 = torch.matmul(x, w1) + b1

    h1 = torch.relu(z1)

    logits = torch.matmul(h1, w2) + b2

    loss = F.cross_entropy(logits, labels)

    print("z1.shape:", z1.shape)
    print("h1.shape:", h1.shape)
    print("logits.shape:", logits.shape)
    print("loss:", loss.item())

    loss.backward()

    print("w1.grad:\n", w1.grad)
    print("b1.grad:\n", b1.grad)
    print("w2.grad:\n", w2.grad)
    print("b2.grad:\n", b2.grad)

    first_w1_grad = w1.grad.clone()

    # Codex 演示代码：重新前向传播会建立一张新的计算图。
    # 第二次 backward 不会覆盖旧梯度，而是累加到 w1.grad。
    z1_second = torch.matmul(x, w1) + b1
    h1_second = torch.relu(z1_second)
    logits_second = torch.matmul(h1_second, w2) + b2
    loss_second = F.cross_entropy(logits_second, labels)
    loss_second.backward()

    print("first w1.grad:\n", first_w1_grad)
    print("after second backward:\n", w1.grad)
    print(
        "gradient doubled:",
        torch.allclose(w1.grad, 2 * first_w1_grad),
    )

    for parameter in (w1, b1, w2, b2):
        parameter.grad.zero_()

    print("w1.grad after zero:\n", w1.grad)
    print("b1.grad after zero:\n", b1.grad)
    print("w2.grad after zero:\n", w2.grad)
    print("b2.grad after zero:\n", b2.grad)

    # 重新前向传播，得到新的计算图
    z1_update = x @ w1 + b1
    h1_update = torch.relu(z1_update)
    logits_update = h1_update @ w2 + b2
    loss_before_update = F.cross_entropy(logits_update, labels)

    # 重新计算梯度
    loss_before_update.backward()

    learning_rate = 0.1

    # 更新参数，但不让更新操作进入计算图
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        b1 -= learning_rate * b1.grad
        w2 -= learning_rate * w2.grad
        b2 -= learning_rate * b2.grad

    # 更新后再次前向传播，检查损失
    with torch.no_grad():
        z1_after = x @ w1 + b1
        h1_after = torch.relu(z1_after)
        logits_after = h1_after @ w2 + b2
        loss_after_update = F.cross_entropy(logits_after, labels)

    print("loss before update:", loss_before_update.item())
    print("loss after update:", loss_after_update.item())


if __name__ == "__main__":
    main()
