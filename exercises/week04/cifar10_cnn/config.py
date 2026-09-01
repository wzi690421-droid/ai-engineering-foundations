from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str = "small_cnn_baseline"
    model_name: str = "small_cnn"
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 0.001
    target_epochs: int = 20
    validation_size: int = 5_000
    use_data_augmentation: bool = True
    random_crop_padding: int = 4
    horizontal_flip_probability: float = 0.5
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
