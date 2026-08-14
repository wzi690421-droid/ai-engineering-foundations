import numpy as np


def linear_forward(x, w, b):
    return np.dot(x, w) + b


def relu(x):
    return np.maximum(0, x)


def stable_softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probabilities


def cross_entropy_from_logits(logits, labels):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_sum = np.sum(np.exp(logits), axis=1)
    ln_exp_sum = np.log(exp_sum)
    n_samples = ln_exp_sum.shape[0]
    like = logits[np.arange(n_samples), labels]
    log_likelihood = ln_exp_sum - like
    loss = np.mean(log_likelihood)
    return loss


def two_layer_forward(x, w1, b1, w2, b2,negative_slope = 0.0):
    z1 = linear_forward(x, w1, b1)

    h1 = np.where( z1>0, z1, z1 * negative_slope )

    logits = linear_forward(h1, w2, b2)

    probabilities = stable_softmax(logits)

    cache = {
        "x": x,
        "z1": z1,
        "h1": h1,
        "logits": logits,
        "probabilities": probabilities,
        "negative_slope": negative_slope,
    }

    return probabilities, cache


def two_layer_backward(labels, cache, w2):
    probabilities = cache["probabilities"]
    negative_slope = cache["negative_slope"]
    z1 = cache["z1"]

    n_samples = probabilities.shape[0]
    dlogits = probabilities.copy()
    dlogits[np.arange(n_samples), labels] -= 1
    dlogits /= n_samples

    dw2 = np.dot(cache["h1"].T, dlogits)
    db2 = np.sum(dlogits, axis=0)

    dh1 = np.dot(dlogits, w2.T)
    dz1 = np.where( z1>0, dh1, dh1 * negative_slope )

    dw1 = np.dot(cache["x"].T, dz1)
    db1 = np.sum(dz1, axis=0)

    gradients = {
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2,
    }
    return gradients


def train_step(
    x,
    labels,
    parameters,
    learning_rate,
    negative_slope=0.0,
):
    _, cache = two_layer_forward(
        x,
        parameters["w1"],
        parameters["b1"],
        parameters["w2"],
        parameters["b2"],
        negative_slope=negative_slope,
    )

    loss = cross_entropy_from_logits(cache["logits"], labels)
    gradients = two_layer_backward(labels, cache, parameters["w2"])

    parameters["w1"] -= learning_rate * gradients["dw1"]
    parameters["b1"] -= learning_rate * gradients["db1"]
    parameters["w2"] -= learning_rate * gradients["dw2"]
    parameters["b2"] -= learning_rate * gradients["db2"]

    return loss
