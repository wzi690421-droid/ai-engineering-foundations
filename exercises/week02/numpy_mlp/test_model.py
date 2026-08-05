import numpy as np

from model import (
    cross_entropy_from_logits,
    linear_forward,
    relu,
    softmax,
    two_layer_backward,
    two_layer_forward,
)


def make_inputs():
    x = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    w = np.array([
        [1.0, 0.0, -1.0],
        [0.0, 2.0, 1.0],
    ])
    b = np.array([1.0, -1.0, 0.0])
    return x, w, b


def assert_value_error(x, w, b, expected_message):
    try:
        linear_forward(x, w, b)
    except ValueError as error:
        assert str(error) == expected_message
    else:
        raise AssertionError(f"expected ValueError: {expected_message}")


def test_basic_forward():
    x, w, b = make_inputs()
    expected = np.array([
        [2.0, 3.0, 1.0],
        [4.0, 7.0, 1.0],
    ])

    actual = linear_forward(x, w, b)

    np.testing.assert_allclose(actual, expected)


def test_single_sample():
    _, w, b = make_inputs()
    x = np.array([[5.0, 6.0]])

    actual = linear_forward(x, w, b)

    assert actual.shape == (1, 3)
    np.testing.assert_allclose(actual, np.array([[6.0, 11.0, 1.0]]))


def test_invalid_shapes():
    x, w, b = make_inputs()

    assert_value_error(
        np.array([1.0, 2.0]), w, b, "x must be a 2D array"
    )
    assert_value_error(
        x, np.array([1.0, 2.0]), b, "w must be a 2D array"
    )
    assert_value_error(
        x, w, np.array([[1.0, -1.0, 0.0]]), "b must be a 1D array"
    )
    assert_value_error(
        x, np.ones((3, 3)), b, "x and w shapes are incompatible"
    )
    assert_value_error(
        x, w, np.array([1.0, -1.0]), "b size must match w output size"
    )


def test_inputs_unchanged():
    x, w, b = make_inputs()
    x_before = x.copy()
    w_before = w.copy()
    b_before = b.copy()

    linear_forward(x, w, b)

    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(w, w_before)
    np.testing.assert_array_equal(b, b_before)


def test_day2_math():
    values = np.array([[-2.0, 3.0, 0.0], [5.0, -1.0, 4.0]])
    np.testing.assert_array_equal(
        relu(values),
        np.array([[0.0, 3.0, 0.0], [5.0, 0.0, 4.0]]),
    )

    logits = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
    probabilities = softmax(logits)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))
    assert np.isfinite(probabilities).all()

    normal_loss = cross_entropy_from_logits(
        np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]]),
        np.array([0, 2]),
    )
    np.testing.assert_allclose(normal_loss, 0.4076059644443804)

    extreme_loss = cross_entropy_from_logits(
        np.array([[1000.0, 0.0, -1000.0]]),
        np.array([2]),
    )
    np.testing.assert_allclose(extreme_loss, 2000.0)


def test_two_layer_forward():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    w1 = np.array([[1.0, -1.0, 0.5], [0.0, 2.0, -1.0]])
    b1 = np.array([0.0, 1.0, 0.5])
    w2 = np.array([[1.0, -1.0], [0.5, 1.0], [-1.0, 0.5]])
    b2 = np.array([0.2, -0.2])

    probabilities, cache = two_layer_forward(x, w1, b1, w2, b2)

    assert cache["z1"].shape == (2, 3)
    assert cache["h1"].shape == (2, 3)
    assert cache["logits"].shape == (2, 2)
    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(
        probabilities,
        np.array([
            [0.598687660112452, 0.401312339887548],
            [0.967704535301549, 0.032295464698451],
        ]),
    )

    gradients = two_layer_backward(np.array([0, 1]), cache, w2)
    expected_shapes = {
        "dw1": w1.shape,
        "db1": b1.shape,
        "dw2": w2.shape,
        "db2": b2.shape,
    }
    for name, expected_shape in expected_shapes.items():
        assert gradients[name].shape == expected_shape

    np.testing.assert_allclose(
        gradients["dw2"],
        np.array([
            [1.250900633682399, -1.250900633682399],
            [2.100488932125654, -2.100488932125654],
            [0.0, 0.0],
        ]),
    )


def main():
    test_basic_forward()
    test_single_sample()
    test_invalid_shapes()
    test_inputs_unchanged()
    test_day2_math()
    test_two_layer_forward()
    print("Week 2 Day 1-4 model functions passed")


if __name__ == "__main__":
    main()
