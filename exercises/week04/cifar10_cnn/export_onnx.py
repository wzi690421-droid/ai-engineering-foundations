import onnx
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
from model import SmallCNN


FROZEN_RUN_ID = "run_20260812_165446"

def main() -> None:
    current_dir = Path(__file__).resolve().parent

    best_model_path = (
        current_dir
        / "experiments"
        / "small_cnn_augmentation_on"
        / FROZEN_RUN_ID
        / "checkpoints"
        / "best_model.pt"
    )


    print("current directory:", current_dir)
    print("best model path:", best_model_path)
    print("model exists:", best_model_path.is_file())

    # 先根据Python代码创建一个随机参数的SmallCNN结构。
    model = SmallCNN()

    # 从文件中读取训练好的参数字典。
    state_dict = torch.load(
        best_model_path,
        map_location="cpu",
        weights_only=True,
    )

    # 把训练好的参数填入SmallCNN的对应层。
    load_result = model.load_state_dict(state_dict)

    # 导出的是推理模型，因此切换到评估模式。
    model.eval()

    print("load result:", load_result)
    print("training mode:", model.training)

    torch.manual_seed(42)

    example_images = torch.randn(
        1,
        3,
        32,
        32,
        dtype=torch.float32,
    )

    # 当前只执行前向推理，不建立反向传播计算图。
    with torch.inference_mode():
        example_logits = model(example_images)

    print("example input shape:", example_images.shape)
    print("example input dtype:", example_images.dtype)
    print("example output shape:", example_logits.shape)
    print("example output dtype:", example_logits.dtype)

    onnx_path = (
        current_dir
        / "experiments"
        / "small_cnn_augmentation_on"
        / FROZEN_RUN_ID
        / "exports"
        / "small_cnn_dynamic_batch.onnx"
    )

    onnx_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 创建一个名为 batch 的动态维度，最少允许输入1张图片。
    batch_dimension = torch.export.Dim(
        "batch",
        min=1,
    )

    onnx_program = torch.onnx.export(
        model=model,
        args=(example_images,),
        f=onnx_path,
        input_names=["images"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
        external_data=False,
        dynamic_shapes={
            "images": {
                0: batch_dimension,
            }
        }
    )

    onnx_model = onnx.load(onnx_path)

    onnx.checker.check_model(onnx_model)

    print("ONNX graph nodes:")

    for node_index, node in enumerate(onnx_model.graph.node):
        print(
            f"{node_index}: "
            f"{node.op_type} | "
            f"{list(node.input)} -> {list(node.output)}"
        )

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"]
    )

    print("ORT input name:", session.get_inputs()[0].name)
    print("ORT input shape:", session.get_inputs()[0].shape)
    print("ORT input type:", session.get_inputs()[0].type)

    print("ORT output name:", session.get_outputs()[0].name)
    print("ORT output shape:", session.get_outputs()[0].shape)
    print("ORT output type:", session.get_outputs()[0].type)

    ort_outputs = session.run(
        ["logits"],
        {
            "images": example_images.numpy(),
        }
    )

    ort_logits = ort_outputs[0]

    pytorch_logits = example_logits.numpy()

    maximum_absolute_error = np.max(
        np.abs(pytorch_logits - ort_logits)
    )

    pytorch_prediction = int(
        np.argmax(pytorch_logits, axis=1)[0]
    )
    ort_prediction = int(
        np.argmax(ort_logits, axis=1)[0]
    )

    # 不要求二进制逐位相同，但要求数值误差在浮点容忍范围内。
    np.testing.assert_allclose(
        pytorch_logits,
        ort_logits,
        rtol=1e-5,
        atol=1e-5,
    )

    print("PyTorch logits:", pytorch_logits)
    print("ORT logits:", ort_logits)
    print("maximum absolute error:", maximum_absolute_error)
    print("PyTorch prediction:", pytorch_prediction)
    print("ORT prediction:", ort_prediction)
    print("PyTorch/ORT alignment: passed")

    print("ONNX model check: passed")
    print("ONNX path:", onnx_path)
    print("ONNX exists:", onnx_path.is_file())
    print("ONNX size:", onnx_path.stat().st_size, "bytes")
    print("export result type:", type(onnx_program))

    # 验证动态ONNX能够处理不同的batch，并检查它与PyTorch的输出是否一致。
    for batch_size in [1, 8, 64]:
        test_images = torch.randn(
            batch_size,
            3,
            32,
            32,
            dtype=torch.float32,
        )

        # 两个推理后端必须接收完全相同的输入，比较结果才有意义。
        with torch.inference_mode():
            test_pytorch_logits = model(test_images).numpy()

        test_ort_logits = session.run(
            ["logits"],
            {
                "images": test_images.numpy(),
            },
        )[0]

        test_maximum_error = np.max(
            np.abs(test_pytorch_logits - test_ort_logits)
        )

        # 输出不要求逐位相同，但全部元素都必须处于允许的浮点误差内。
        np.testing.assert_allclose(
            test_pytorch_logits,
            test_ort_logits,
            rtol=1e-5,
            atol=1e-5,
        )

        print(
            f"batch={batch_size}, "
            f"PyTorch shape={test_pytorch_logits.shape}, "
            f"ORT shape={test_ort_logits.shape}, "
            f"maximum error={test_maximum_error:.8f}, "
            "alignment=passed"
        )

if __name__ == "__main__":
    main()
