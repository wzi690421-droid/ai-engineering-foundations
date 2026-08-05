import numpy as np

def linear_forward(x, w, b):
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    if w.ndim != 2:
        raise ValueError("w must be a 2D array")
    if b.ndim != 1:
        raise ValueError("b must be a 1D array")

    if x.shape[1] != w.shape[0]:
        raise ValueError("x and w shapes are incompatible")

    if b.shape[0] != w.shape[1]:
        raise ValueError("b size must match w output size")

    matrix_result = x @ w
    y = matrix_result + b
    return y

def relu(x):
    return np.maximum(0, x)

def softmax(logits):
    row_max = np.max(logits, axis=1, keepdims=True)
    shifted_logits = logits - row_max

    exp_values = np.exp(shifted_logits)
    row_sums = np.sum(exp_values, axis=1, keepdims=True)

    probabilities = exp_values / row_sums
    return probabilities

def cross_entropy(probabilities, labels):
    sample_count = probabilities.shape[0]

    correct_class_probabilities = probabilities[
        np.arange(sample_count), labels
    ]

    safe_probabilities = np.clip(
        correct_class_probabilities, 1e-12, 1.0
    )

    sample_losses = -np.log(safe_probabilities)
    loss = np.mean(sample_losses)
    return loss

def classification_loss(logits, labels):
    probabilities = softmax(logits)
    loss = cross_entropy(probabilities, labels)
    return probabilities, loss

def cross_entropy_from_logits(logits, labels):
    row_max = np.max(logits, axis=1, keepdims=True)
    shifted_logits = logits - row_max

    sample_count = logits.shape[0]

    log_normalizers = np.log(
        np.sum(np.exp(shifted_logits), axis=1)
    )

    correct_shifted_logits = shifted_logits[
        np.arange(sample_count), labels
    ]

    sample_losses = log_normalizers - correct_shifted_logits
    loss = np.mean(sample_losses)

    return loss
