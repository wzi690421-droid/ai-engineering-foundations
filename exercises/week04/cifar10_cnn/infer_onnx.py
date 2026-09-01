import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np
import onnxruntime as ort

FROZEN_RUN_ID = "run_20260812_165446"
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

def parse_arguments() -> argparse.Namespace:
    # 创建命令行参数解析器。
    parser = argparse.ArgumentParser(
        description="使用ONNX Runtime执行图片分类。",
    )

    # 至少接收一个图片路径，也允许一次传入多个路径。
    parser.add_argument(
        "image_paths",
        type=Path,
        nargs="+",
        help="需要分类的一张或多张图片路径。",
    )

    # 解析终端输入并返回结果。
    return parser.parse_args()

def preprocess_image(
    image_path: Path,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    # 打开图片。with结束后，图片文件会自动关闭。
    with Image.open(image_path) as image:
        # 不管原图是灰度图、RGBA还是其他格式，都统一转成RGB三通道。
        rgb_image = image.convert("RGB")

        # SmallCNN训练时使用32×32图片，因此推理输入也必须是32×32。
        resized_image = rgb_image.resize(
            (32, 32),
            resample=Image.Resampling.BILINEAR,
        )

        # PIL图片转换成NumPy数组。
        # 当前形状是[32, 32, 3]，数值仍然处于0～255。
        image_array = np.asarray(
            resized_image,
            dtype=np.float32,
        )

    # 对应Torchvision的ToTensor数值转换：
    # 把像素从0～255转换到0～1。
    image_array = image_array / 255.0

    # NumPy图片排列是[H, W, C]。
    # 模型要求[C, H, W]，所以把通道维移动到最前面。
    channel_first_image = np.transpose(
        image_array,
        (2, 0, 1),
    )

    # mean和std原本形状是[3]。
    # 改为[3,1,1]后，三个通道才能分别使用各自的均值和标准差。
    channel_mean = mean.reshape(3, 1, 1)
    channel_std = std.reshape(3, 1, 1)

    # 对应训练阶段的Normalize：
    # normalized = (像素值 - 均值) / 标准差。
    normalized_image = (
        channel_first_image - channel_mean
    ) / channel_std

    # 目前形状是[3,32,32]，只表示一张图片。
    # 增加batch维后变成[1,3,32,32]。
    batched_image = np.expand_dims(
        normalized_image,
        axis=0,
    )

    # transpose可能产生不连续的内存布局。
    # ONNX Runtime需要float32输入，这里同时保证类型和连续内存。
    return np.ascontiguousarray(
        batched_image,
        dtype=np.float32,
    )

def main() -> None:
    arguments = parse_arguments()

    current_dir = Path(__file__).resolve().parent

    run_dir = (
        current_dir
        / "experiments"
        / "small_cnn_augmentation_on"
        / FROZEN_RUN_ID
    )

    onnx_path = (
        run_dir
        / "exports"
        / "small_cnn_dynamic_batch.onnx"
    )

    config_path = run_dir / "config.json"

    # 从冻结实验配置读取训练时使用的归一化参数。
    # 这样推理端不会手动复制另一套mean和std。
    with config_path.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = json.load(config_file)

    normalization = config["normalization"]

    mean = np.asarray(
        normalization["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        normalization["std"],
        dtype=np.float32,
    )

    # 每张图片先独立预处理，单个结果形状都是[1,3,32,32]。
    preprocessed_images = []
    for image_path in arguments.image_paths:
        single_image_tensor = preprocess_image(
            image_path,
            mean,
            std,
        )
        preprocessed_images.append(single_image_tensor)

    # 沿第0维拼接，N个[1,3,32,32]组成[N,3,32,32]的动态batch。
    input_tensor = np.concatenate(
        preprocessed_images,
        axis=0,
    )

    print("image paths:", arguments.image_paths)
    print("image count:", len(arguments.image_paths))
    print("input tensor shape:", input_tensor.shape)
    print("input tensor dtype:", input_tensor.dtype)
    print(
        "continuous memory:",
        input_tensor.flags["C_CONTIGUOUS"],
    )
    print("minimum value:", input_tensor.min())
    print("maximum value:", input_tensor.max())

    # 创建ONNX Runtime推理会话。
    # 这里没有创建SmallCNN，也没有读取best_model.pt。
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    # 取得ONNX计算图定义的输入和输出信息。
    input_information = session.get_inputs()[0]
    output_information = session.get_outputs()[0]

    # 从模型信息中读取名称，而不是手动写死"images"和"logits"。
    input_name = input_information.name
    output_name = output_information.name

    # 执行一次ONNX Runtime前向推理。
    # 第一个参数表示需要取得哪些输出。
    # 第二个参数是输入名称与输入张量之间的映射。
    runtime_outputs = session.run(
        [output_name],
        {
            input_name: input_tensor,
        },
    )

    # session.run总是返回一个列表。
    # 当前模型只有一个输出，因此取列表中的第0项。
    logits = runtime_outputs[0]

    # 每张图片分别减去自己的最大logit，防止exp计算时数值溢出。
    shifted_logits = logits - np.max(
        logits,
        axis=1,
        keepdims=True,
    )

    # Softmax把10个原始分数转换成总和为1的类别概率。
    exponentials = np.exp(shifted_logits)
    probabilities = exponentials / np.sum(
        exponentials,
        axis=1,
        keepdims=True,
    )

    # 每一行分别排序；反转类别维后，保留每张图片概率最高的三个下标。
    sorted_indices = np.argsort(
        probabilities,
        axis=1,
    )
    top3_indices = sorted_indices[:, ::-1][:, :3]

    print("runtime output count:", len(runtime_outputs))
    print("logits shape:", logits.shape)
    print("logits dtype:", logits.dtype)
    print("logits:", logits)
    print("probabilities:", probabilities)
    print("probability sums:", probabilities.sum(axis=1))

    # 根据图片所在行，分别读取该图片的三个最高概率类别。
    for image_index, image_path in enumerate(arguments.image_paths):
        print(f"image {image_index}: {image_path}")
        print("top-3 predictions:")

        for class_index in top3_indices[image_index]:
            print(
                f"  {CLASS_NAMES[class_index]}: "
                f"{probabilities[image_index, class_index]:.2%}"
            )
    print("ONNX path:", onnx_path)
    print("providers:", session.get_providers())
    print("input name:", session.get_inputs()[0].name)
    print("input shape:", session.get_inputs()[0].shape)
    print("output name:", session.get_outputs()[0].name)
    print("output shape:", session.get_outputs()[0].shape)
    print("mean:", mean)
    print("std:", std)


if __name__ == "__main__":
    main()
