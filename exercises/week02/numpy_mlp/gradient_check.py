import numpy as np

from model import (
    cross_entropy_from_logits,
    two_layer_backward,
    two_layer_forward,
)


def make_example():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = np.array([0, 1])
    parameters = {
        "w1": np.array([[1.0, -1.0, 0.5], [0.0, 2.0, -1.0]]),
        "b1": np.array([0.0, 1.0, 0.5]),
        "w2": np.array([[1.0, -1.0], [0.5, 1.0], [-1.0, 0.5]]),
        "b2": np.array([0.2, -0.2]),
    }
    return x, labels, parameters


def calculate_loss(x, labels, parameters):
    _, cache = two_layer_forward(
        x,
        parameters["w1"],
        parameters["b1"],
        parameters["w2"],
        parameters["b2"],
    )
    return cross_entropy_from_logits(cache["logits"], labels)


def numerical_gradient(x, labels, parameters, parameter_name, epsilon=1e-5):
    parameter = parameters[parameter_name]
    gradient = np.zeros_like(parameter)

    for index in np.ndindex(parameter.shape):
        original_value = parameter[index]

        parameter[index] = original_value + epsilon
        loss_plus = calculate_loss(x, labels, parameters)

        parameter[index] = original_value - epsilon
        loss_minus = calculate_loss(x, labels, parameters)

        parameter[index] = original_value
        gradient[index] = (loss_plus - loss_minus) / (2.0 * epsilon)

    return gradient


def maximum_relative_error(analytic, numerical):
    denominator = np.maximum(
        1e-12,
        np.abs(analytic) + np.abs(numerical),
    )
    return np.max(np.abs(analytic - numerical) / denominator)


def main():
    x, labels, parameters = make_example()
    _, cache = two_layer_forward(
        x,
        parameters["w1"],
        parameters["b1"],
        parameters["w2"],
        parameters["b2"],
    )
    analytic_gradients = two_layer_backward(labels, cache, parameters["w2"])

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
