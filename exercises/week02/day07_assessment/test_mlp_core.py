import numpy as np

from mlp_core import (
    cross_entropy_from_logits,
    linear_forward,
    relu,
    stable_softmax,
    train_step,
    two_layer_backward,
    two_layer_forward,
)


def test_linear_and_relu():
    x = np.array([[1.0, 2.0], [-1.0, 3.0]])
    w = np.array([[2.0, -1.0, 0.5], [1.0, 2.0, -2.0]])
    b = np.array([0.5, -0.5, 1.0])

    actual = linear_forward(x, w, b)
    expected = np.array([[4.5, 2.5, -2.5], [1.5, 6.5, -5.5]])

    assert np.allclose(actual, expected)
    assert np.array_equal(
        relu(np.array([[-2.0, 0.0, 3.0]])),
        np.array([[0.0, 0.0, 3.0]]),
    )


def test_stable_softmax():
    logits = np.array([
        [1000.0, 1001.0, 999.0],
        [-1000.0, -1002.0, -999.0],
    ])
    probabilities = stable_softmax(logits)

    assert probabilities.shape == (2, 3)
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0)
    assert np.allclose(probabilities.sum(axis=1), np.ones(2))
    assert np.array_equal(probabilities.argmax(axis=1), np.array([1, 2]))


def test_cross_entropy_is_stable_and_per_sample():
    logits = np.array([
        [1000.0, 999.0, 998.0],
        [-1000.0, -999.0, -998.0],
        [5.0, -3.0, 2.0],
        [0.0, 0.0, 0.0],
    ])
    labels = np.array([0, 2, 1, 1])

    row_max = np.max(logits, axis=1)
    expected_losses = (
        row_max
        + np.log(np.sum(np.exp(logits - row_max[:, None]), axis=1))
        - logits[np.arange(logits.shape[0]), labels]
    )
    actual = cross_entropy_from_logits(logits, labels)

    assert np.isscalar(actual)
    assert np.isfinite(actual)
    assert np.allclose(actual, np.mean(expected_losses))


def test_two_layer_forward_and_cache():
    x = np.array([[1.0, -2.0], [0.5, 3.0], [-1.0, 1.0]])
    w1 = np.array([[1.0, -1.0, 0.5], [2.0, 1.0, -0.5]])
    b1 = np.array([0.0, 0.5, 1.0])
    w2 = np.array([[1.0, -2.0], [0.5, 1.0], [-1.0, 0.25]])
    b2 = np.array([0.25, -0.5])

    x_before = x.copy()
    probabilities, cache = two_layer_forward(x, w1, b1, w2, b2)

    expected_z1 = x @ w1 + b1
    expected_h1 = np.maximum(0.0, expected_z1)
    expected_logits = expected_h1 @ w2 + b2

    assert probabilities.shape == (3, 2)
    assert np.allclose(probabilities.sum(axis=1), np.ones(3))
    assert set(cache) >= {"x", "z1", "h1", "logits", "probabilities"}
    assert np.allclose(cache["z1"], expected_z1)
    assert np.allclose(cache["h1"], expected_h1)
    assert np.allclose(cache["logits"], expected_logits)
    assert np.allclose(cache["probabilities"], probabilities)
    assert np.array_equal(x, x_before)


def numerical_gradient(parameter, calculate_loss, epsilon=1e-5):
    gradient = np.zeros_like(parameter)

    for index in np.ndindex(parameter.shape):
        original = parameter[index]

        parameter[index] = original + epsilon
        loss_plus = calculate_loss()

        parameter[index] = original - epsilon
        loss_minus = calculate_loss()

        parameter[index] = original
        gradient[index] = (loss_plus - loss_minus) / (2.0 * epsilon)

    return gradient


def relative_error(actual, expected):
    denominator = np.maximum(1e-8, np.abs(actual) + np.abs(expected))
    return np.max(np.abs(actual - expected) / denominator)


def test_two_layer_backward_with_numerical_gradients():
    x = np.array([[0.2, -0.4], [1.0, 0.5], [-0.3, 0.8]])
    labels = np.array([0, 1, 0])
    parameters = {
        "w1": np.array([[0.6, -0.2, 0.4], [-0.5, 0.7, 0.3]]),
        "b1": np.array([0.2, 0.1, -0.4]),
        "w2": np.array([[0.2, -0.4], [0.5, 0.3], [-0.2, 0.6]]),
        "b2": np.array([0.1, -0.2]),
    }

    _, cache = two_layer_forward(
        x,
        parameters["w1"],
        parameters["b1"],
        parameters["w2"],
        parameters["b2"],
    )
    analytic = two_layer_backward(labels, cache, parameters["w2"])

    def calculate_loss():
        _, current_cache = two_layer_forward(
            x,
            parameters["w1"],
            parameters["b1"],
            parameters["w2"],
            parameters["b2"],
        )
        return cross_entropy_from_logits(current_cache["logits"], labels)

    for parameter_name, gradient_name in (
        ("w1", "dw1"),
        ("b1", "db1"),
        ("w2", "dw2"),
        ("b2", "db2"),
    ):
        numerical = numerical_gradient(
            parameters[parameter_name],
            calculate_loss,
        )
        assert analytic[gradient_name].shape == parameters[parameter_name].shape
        assert relative_error(analytic[gradient_name], numerical) < 1e-6


def test_leaky_relu_forward_and_backward():
    x = np.array([[0.2, -0.4], [1.0, 0.5], [-0.3, 0.8]])
    labels = np.array([0, 1, 0])
    negative_slope = 0.1
    parameters = {
        "w1": np.array([[0.6, -0.2, 0.4], [-0.5, 0.7, 0.3]]),
        "b1": np.array([0.2, 0.1, -0.4]),
        "w2": np.array([[0.2, -0.4], [0.5, 0.3], [-0.2, 0.6]]),
        "b2": np.array([0.1, -0.2]),
    }

    _, cache = two_layer_forward(
        x,
        parameters["w1"],
        parameters["b1"],
        parameters["w2"],
        parameters["b2"],
        negative_slope=negative_slope,
    )
    expected_h1 = np.where(
        cache["z1"] > 0.0,
        cache["z1"],
        negative_slope * cache["z1"],
    )
    assert np.allclose(cache["h1"], expected_h1)
    assert cache["negative_slope"] == negative_slope

    analytic = two_layer_backward(labels, cache, parameters["w2"])

    def calculate_loss():
        _, current_cache = two_layer_forward(
            x,
            parameters["w1"],
            parameters["b1"],
            parameters["w2"],
            parameters["b2"],
            negative_slope=negative_slope,
        )
        return cross_entropy_from_logits(current_cache["logits"], labels)

    for parameter_name, gradient_name in (
        ("w1", "dw1"),
        ("b1", "db1"),
        ("w2", "dw2"),
        ("b2", "db2"),
    ):
        numerical = numerical_gradient(
            parameters[parameter_name],
            calculate_loss,
        )
        assert relative_error(analytic[gradient_name], numerical) < 1e-6


def test_train_step_updates_parameters_with_computed_gradients():
    x = np.array([[0.2, -0.4], [1.0, 0.5], [-0.3, 0.8]])
    labels = np.array([0, 1, 0])
    learning_rate = 0.05
    negative_slope = 0.1
    parameters = {
        "w1": np.array([[0.6, -0.2, 0.4], [-0.5, 0.7, 0.3]]),
        "b1": np.array([0.2, 0.1, -0.4]),
        "w2": np.array([[0.2, -0.4], [0.5, 0.3], [-0.2, 0.6]]),
        "b2": np.array([0.1, -0.2]),
    }

    parameters_before = {
        name: value.copy()
        for name, value in parameters.items()
    }
    _, cache = two_layer_forward(
        x,
        parameters_before["w1"],
        parameters_before["b1"],
        parameters_before["w2"],
        parameters_before["b2"],
        negative_slope=negative_slope,
    )
    expected_loss = cross_entropy_from_logits(cache["logits"], labels)
    gradients = two_layer_backward(labels, cache, parameters_before["w2"])

    actual_loss = train_step(
        x,
        labels,
        parameters,
        learning_rate,
        negative_slope=negative_slope,
    )

    assert np.allclose(actual_loss, expected_loss)
    for parameter_name, gradient_name in (
        ("w1", "dw1"),
        ("b1", "db1"),
        ("w2", "dw2"),
        ("b2", "db2"),
    ):
        expected = (
            parameters_before[parameter_name]
            - learning_rate * gradients[gradient_name]
        )
        assert np.allclose(parameters[parameter_name], expected)


def main():
    test_linear_and_relu()
    test_stable_softmax()
    test_cross_entropy_is_stable_and_per_sample()
    test_two_layer_forward_and_cache()
    test_two_layer_backward_with_numerical_gradients()
    test_leaky_relu_forward_and_backward()
    test_train_step_updates_parameters_with_computed_gradients()
    print("Week 02 Day 07 coding assessment passed")


if __name__ == "__main__":
    main()
