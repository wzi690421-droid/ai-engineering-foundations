import numpy as np

from gradient_check import (
    make_example,
    maximum_relative_error,
    numerical_gradient,
)
from model import two_layer_forward


def buggy_two_layer_backward(labels, cache, w2):
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

    dz1 = dh1

    dw1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0)

    return {
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2,
    }


def main():
    x, labels, parameters = make_example()
    _, cache = two_layer_forward(
        x,
        parameters["w1"],
        parameters["b1"],
        parameters["w2"],
        parameters["b2"],
    )
    analytic_gradients = buggy_two_layer_backward(
        labels,
        cache,
        parameters["w2"],
    )

    gradient_names = {
        "w1": "dw1",
        "b1": "db1",
        "w2": "dw2",
        "b2": "db2",
    }

    for parameter_name, gradient_name in gradient_names.items():
        numerical = numerical_gradient(
            x,
            labels,
            parameters,
            parameter_name,
        )
        error = maximum_relative_error(
            analytic_gradients[gradient_name],
            numerical,
        )
        status = "PASS" if error < 1e-7 else "FAIL"
        print(f"{gradient_name}: relative_error={error:.3e} {status}")


if __name__ == "__main__":
    main()
