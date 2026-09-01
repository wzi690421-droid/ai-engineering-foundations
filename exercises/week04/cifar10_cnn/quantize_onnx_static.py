import numpy as np
import json

import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import (
    quant_pre_process,
)
from pathlib import Path

import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split

from data import (
    TransformedSubset,
    create_evaluation_transform,
    create_train_validation_loaders,
)

class Cifar10CalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        data_loader: DataLoader,
        input_name: str,
    ):
        # 保存DataLoader，以便需要时能够重新开始读取。
        self.data_loader = data_loader

        # 当前模型的输入名称是images。
        # 返回校准数据时，字典的键必须与它一致。
        self.input_name = input_name

        # iterator记录当前已经读取到第几个batch。
        self.iterator = iter(data_loader)

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            # images是[N, 3, 32, 32]；
            # labels只用于分类监督，校准不需要标签。
            images, _ = next(self.iterator)
        except StopIteration:
            # 返回None，告诉ORT校准数据已经读完。
            return None

        # PyTorch Tensor转换为连续的NumPy float32数组。
        input_batch = (
            images
            .contiguous()
            .numpy()
            .astype(np.float32, copy=False)
        )

        # 格式与ORT的session.run输入相同：
        # {"images": 一个NCHW数组}
        return {
            self.input_name: input_batch
        }

    def rewind(self) -> None:
        # 重新创建迭代器，使读取位置回到第一批。
        self.iterator = iter(self.data_loader)

def create_calibration_data_loader(
    data_dir: Path,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    seed: int,
    validation_size: int,
    calibration_size: int = 512,
    batch_size: int = 32,
) -> DataLoader:
    # 校准集来自CIFAR-10官方训练部分。
    # transform暂时设为None，后面只给选中的图片应用固定预处理。
    full_train_dataset = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=None,
    )

    total_size = len(full_train_dataset)
    train_size = total_size - validation_size

    if not 0 < validation_size < total_size:
        raise ValueError(
            "validation_size must be between 1 "
            "and total_size - 1"
        )

    if not 0 < calibration_size <= train_size:
        raise ValueError(
            "calibration_size must be positive "
            "and cannot exceed train_size"
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive"
        )

    # 使用与原训练工程相同的seed和划分大小，
    # 重建完全相同的训练集/验证集成员。
    split_generator = torch.Generator().manual_seed(seed)

    train_index_subset, _ = random_split(
        full_train_dataset,
        lengths=[train_size, validation_size],
        generator=split_generator,
    )

    # 从训练划分中固定取前calibration_size个下标。
    # 因为划分由seed控制，所以每次运行成员都相同。
    calibration_indices = (
        train_index_subset
        .indices[:calibration_size]
    )

    # 校准要模拟实际推理输入，
    # 所以使用固定ToTensor + Normalize，
    # 不使用RandomCrop和RandomHorizontalFlip。
    calibration_dataset = TransformedSubset(
        dataset=full_train_dataset,
        indices=calibration_indices,
        transform=create_evaluation_transform(
            mean=mean,
            std=std,
        ),
    )

    # 不打乱，保证每次校准的数据顺序也相同。
    return DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

def create_static_int8_model(
    fp32_model_path: Path,
    preprocessed_model_path: Path,
    int8_model_path: Path,
    calibration_reader: Cifar10CalibrationDataReader,
    activation_type: QuantType
) -> None:
    # 保证量化结果目录存在。
    int8_model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 先单独完成模型优化与形状推断。
    # 量化工具知道的张量形状越完整，
    # 越容易识别能够量化的算子和中间张量。
    quant_pre_process(
        input_model=fp32_model_path,
        output_model_path=preprocessed_model_path,

        skip_symbolic_shape=True,
    )

    # 前面的检查已经读取过reader，
    # 真正校准前必须回到第一批。
    calibration_reader.rewind()

    # 使用512张图片运行预处理后的FP32模型，
    # 收集中间激活范围并生成QDQ INT8模型。
    quantize_static(
        model_input=preprocessed_model_path,
        model_output=int8_model_path,
        calibration_data_reader=calibration_reader,

        # QDQ会在张量边界插入
        # QuantizeLinear和DequantizeLinear节点。
        quant_format=QuantFormat.QDQ,

        # 激活和权重都使用有符号INT8。
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,

        # 当前先使用最直观的最小值/最大值校准。
        calibrate_method=CalibrationMethod.MinMax,

        # 当前一个权重张量共用一组量化参数。
        # 如果精度下降明显，后面再实验per-channel。
        per_channel=False,

        # 使用完整INT8范围，暂不缩减为7位有效范围。
        reduce_range=False,
    )


def evaluate_onnx_models(
    fp32_model_path: Path,
    int8_model_path: Path,
    data_loader: DataLoader,
) -> dict[str, float | int]:
    fp32_session = ort.InferenceSession(
        fp32_model_path,
        providers=["CPUExecutionProvider"],
    )

    int8_session = ort.InferenceSession(
        int8_model_path,
        providers=["CPUExecutionProvider"],
    )

    fp32_input_name = (
        fp32_session
        .get_inputs()[0]
        .name
    )

    int8_input_name = (
        int8_session
        .get_inputs()[0]
        .name
    )

    total_samples = 0
    fp32_correct = 0
    int8_correct = 0
    same_predictions = 0

    # A：FP32正确，但INT8错误。
    fp32_correct_int8_wrong = 0

    # B：FP32错误，但INT8正确。
    fp32_wrong_int8_correct = 0

    # C：两个模型都错误，而且预测成不同的错误类别。
    both_wrong_different_prediction = 0

    absolute_error_sum = 0.0
    logit_element_count = 0
    maximum_absolute_error = 0.0

    for images, labels in data_loader:
        input_batch = (
            images
            .contiguous()
            .numpy()
            .astype(np.float32, copy=False)
        )

        labels_array = labels.numpy()

        fp32_logits = fp32_session.run(
            None,
            {fp32_input_name: input_batch},
        )[0]

        int8_logits = int8_session.run(
            None,
            {int8_input_name: input_batch},
        )[0]

        fp32_predictions = fp32_logits.argmax(axis=1)
        int8_predictions = int8_logits.argmax(axis=1)

        fp32_is_correct = (
            fp32_predictions == labels_array
        )

        int8_is_correct = (
            int8_predictions == labels_array
        )

        predictions_differ = (
            fp32_predictions != int8_predictions
        )

        batch_absolute_error = np.abs(
            fp32_logits - int8_logits
        )

        total_samples += labels_array.size

        fp32_correct += np.count_nonzero(
            fp32_is_correct
        )

        int8_correct += np.count_nonzero(
            int8_is_correct
        )

        same_predictions += np.count_nonzero(
            ~predictions_differ
        )

        fp32_correct_int8_wrong += np.count_nonzero(
            fp32_is_correct
            & ~int8_is_correct
        )

        fp32_wrong_int8_correct += np.count_nonzero(
            ~fp32_is_correct
            & int8_is_correct
        )

        both_wrong_different_prediction += (
            np.count_nonzero(
                ~fp32_is_correct
                & ~int8_is_correct
                & predictions_differ
            )
        )

        absolute_error_sum += float(
            batch_absolute_error.sum()
        )

        logit_element_count += (
            batch_absolute_error.size
        )

        maximum_absolute_error = max(
            maximum_absolute_error,
            float(batch_absolute_error.max()),
        )

    changed_prediction_count = (
        total_samples - same_predictions
    )

    # 所有预测变化必须正好属于A、B、C三类。
    if (
        fp32_correct_int8_wrong
        + fp32_wrong_int8_correct
        + both_wrong_different_prediction
        != changed_prediction_count
    ):
        raise RuntimeError(
            "prediction transition counts are inconsistent"
        )

    # 准确数量差必须等于A减B。
    if (
        fp32_correct - int8_correct
        != fp32_correct_int8_wrong
        - fp32_wrong_int8_correct
    ):
        raise RuntimeError(
            "accuracy transition counts are inconsistent"
        )

    fp32_accuracy = fp32_correct / total_samples
    int8_accuracy = int8_correct / total_samples

    return {
        "sample_count": total_samples,
        "changed_prediction_count":
            int(changed_prediction_count),
        "fp32_correct_int8_wrong":
            int(fp32_correct_int8_wrong),
        "fp32_wrong_int8_correct":
            int(fp32_wrong_int8_correct),
        "both_wrong_different_prediction":
            int(both_wrong_different_prediction),
        "fp32_accuracy_percent":
            fp32_accuracy * 100.0,
        "int8_accuracy_percent":
            int8_accuracy * 100.0,
        "accuracy_change_percentage_points":
            (int8_accuracy - fp32_accuracy) * 100.0,
        "top1_agreement_percent":
            same_predictions / total_samples * 100.0,
        "mean_absolute_logit_error":
            absolute_error_sum / logit_element_count,
        "maximum_absolute_logit_error":
            maximum_absolute_error,
    }

def main() -> None:
    # 当前Python文件所在目录：cifar10_cnn。
    current_dir = Path(__file__).resolve().parent

    # 仓库根目录：ai-engineering-foundations。
    project_root = Path(__file__).resolve().parents[3]

    # 使用之前已经训练、测试和导出的固定实验。
    run_dir = (
        current_dir
        / "experiments"
        / "small_cnn_augmentation_on"
        / "run_20260812_165446"
    )

    config_path = run_dir / "config.json"

    model_path = (
        run_dir
        / "exports"
        / "small_cnn_dynamic_batch.onnx"
    )

    # 读取训练时保存的配置。
    # 不能在量化脚本中另外手写一份mean和std。
    with config_path.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = json.load(config_file)

    mean = tuple(
        config["normalization"]["mean"]
    )
    std = tuple(
        config["normalization"]["std"]
    )

    # 创建固定的512张校准图片，每批32张。
    calibration_loader = create_calibration_data_loader(
        data_dir=project_root / "data",
        mean=mean,
        std=std,
        seed=config["seed"],
        validation_size=config["validation_size"],
        calibration_size=512,
        batch_size=32,
    )

    # 从ONNX模型本身读取输入名称，避免手写"images"。
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    reader = Cifar10CalibrationDataReader(
        data_loader=calibration_loader,
        input_name=input_name,
    )

    # 读取第一批，检查ORT真正会收到什么。
    first_inputs = reader.get_next()

    if first_inputs is None:
        raise RuntimeError(
            "calibration reader returned no data"
        )

    first_batch = first_inputs[input_name]

    print("input name:", input_name)
    print("first batch shape:", first_batch.shape)
    print("first batch dtype:", first_batch.dtype)
    print("first batch minimum:", first_batch.min())
    print("first batch maximum:", first_batch.max())

    # 第一批已经读取，所以从1开始统计。
    batch_count = 1

    while reader.get_next() is not None:
        batch_count += 1

    print("calibration samples:", len(calibration_loader.dataset))
    print("calibration batches:", batch_count)
    print("reader exhausted:", reader.get_next() is None)

    # 验证rewind以后能够重新读到第一批。
    reader.rewind()
    rewound_inputs = reader.get_next()

    print("rewind works:", rewound_inputs is not None)

    # 统计512张校准图片中的类别数量。
    # 这里使用DataLoader的新迭代器，不影响reader内部的位置。
    class_counts = torch.zeros(
        10,
        dtype=torch.int64,
    )

    for _, labels in calibration_loader:
        class_counts += torch.bincount(
            labels,
            minlength=10,
        )

    print("class counts:", class_counts.tolist())

    quantization_dir = run_dir / "quantization"

    preprocessed_model_path = (
        quantization_dir
        / "small_cnn_fp32_preprocessed.onnx"
    )

    int8_model_path = (
        quantization_dir
        / "small_cnn_int8_qdq.onnx"
    )

    create_static_int8_model(
        fp32_model_path=model_path,
        preprocessed_model_path=preprocessed_model_path,
        int8_model_path=int8_model_path,
        calibration_reader=reader,
    )

    print(
        "preprocessed model:",
        preprocessed_model_path,
    )
    print(
        "INT8 model:",
        int8_model_path,
    )

    fp32_size = model_path.stat().st_size
    int8_size = int8_model_path.stat().st_size

    print("FP32 model bytes:", fp32_size)
    print("INT8 model bytes:", int8_size)
    print(
        "size ratio:",
        int8_size / fp32_size,
    )

        # 使用原训练工程的函数重建同一份5000张验证集。
    _, validation_loader, validation_mean, validation_std = (
        create_train_validation_loaders(
            data_dir=project_root / "data",
            batch_size=128,
            seed=config["seed"],
            validation_size=config["validation_size"],

            # 验证集本身始终使用固定预处理；
            # 此处关闭增强也避免创建无意义的随机训练变换。
            use_data_augmentation=False,
        )
    )

    # 检查重新计算的统计量是否与实验配置一致。
    # 如果不一致，说明划分或预处理流程发生了漂移。
    if not (
        np.allclose(
            validation_mean,
            mean,
            rtol=0.0,
            atol=1e-12,
        )
        and np.allclose(
            validation_std,
            std,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise RuntimeError(
            "validation normalization does not match "
            "the saved experiment configuration"
        )

    validation_metrics = evaluate_onnx_models(
        fp32_model_path=model_path,
        int8_model_path=int8_model_path,
        data_loader=validation_loader,
    )

    print("validation metrics:")

    for metric_name, metric_value in validation_metrics.items():
        print(
            f"  {metric_name}: "
            f"{metric_value}"
        )

    # 把结果保存下来，后续不同量化方案可以直接比较。
    validation_result_path = (
        quantization_dir
        / "validation_metrics.json"
    )

    with validation_result_path.open(
        "w",
        encoding="utf-8",
    ) as result_file:
        json.dump(
            validation_metrics,
            result_file,
            indent=2,
        )

    print(
        "validation result:",
        validation_result_path,
    )

if __name__ == "__main__":
    main()
