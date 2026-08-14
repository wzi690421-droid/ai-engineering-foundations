from pathlib import Path

import torch

from model import TwoLayerClassifier


def main():
    model = TwoLayerClassifier(
        input_dim=2,
        hidden_dim=3,
        num_classes=2,
    )

    x = torch.randn(4, 2)
    logits = model(x)

    print("x.shape:", x.shape)
    print("logits.shape:", logits.shape)
    print("\nregistered parameters:")

    for name, parameter in model.named_parameters():
        print(
            name,
            parameter.shape,
            "requires_grad:",
            parameter.requires_grad,
        )
    print("\nstate_dict:")

    state = model.state_dict()

    for name, tensor in state.items():
        print(name, tensor.shape)

    checkpoint_path = Path(__file__).with_name("two_layer_state.pt")

    # 保存当前模型的参数数值，不保存模型类和 forward() 代码。
    torch.save(model.state_dict(), checkpoint_path)

    # 记录原输出，再故意修改参数，观察模型输出是否随之改变。
    with torch.no_grad():
        original_logits = model(x).clone()
        model.linear1.weight.add_(10.0)
        changed_logits = model(x).clone()

    # 把保存的参数读取并装回相同结构的模型。
    saved_state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(saved_state)

    with torch.no_grad():
        restored_logits = model(x)

    print(
        "changed output equals original:",
        torch.allclose(changed_logits, original_logits),
    )
    print(
        "restored output equals original:",
        torch.allclose(restored_logits, original_logits),
    )


if __name__ == "__main__":
    main()
