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

def two_layer_forward(x, w1, b1, w2, b2):
    z1 = linear_forward(x, w1, b1)
    h1 = relu(z1)
    logits = linear_forward(h1, w2, b2)
    probabilities = softmax(logits)

    cache = {
        "x": x,
        "z1": z1,
        "h1": h1,
        "logits": logits,
        "probabilities": probabilities
    }

    return probabilities, cache

def two_layer_backward(labels, cache, w2):
    x = cache["x"]
    z1 = cache["z1"]
    h1 = cache["h1"]
    probabilities = cache["probabilities"]

    sample_count = labels.shape[0]

    dlogits = probabilities.copy()
    dlogits[np.arange(sample_count), labels] -= 1.0
    dlogits /= sample_count

    dw2 = h1.T @ dlogits
    db2 = np.sum(dlogits, axis=0)
    dh1 = dlogits @ w2.T

    dz1 = dh1 * (z1 > 0)

    dw1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0)

    gradients = {
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2,
    }

    return gradients
