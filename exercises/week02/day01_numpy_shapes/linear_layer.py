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

def main():
    x = np.array([[1.0, 2.0],
                  [3.0, 4.0],])

    w = np.array([[1.0, 0.0,-1.0],
                  [0.0, 2.0,1.0],])

    b = np.array([1.0, -1.0,0.0])

    actual = linear_forward(x, w, b)

    expected = np.array([[2.0, 3.0, 1.0],
                         [4.0, 7.0, 1.0],])

    np.testing.assert_allclose(actual, expected)

    invalid_x = np.array([1.0, 2.0])
    try:
        linear_forward(invalid_x, w, b)
    except ValueError as error:
        assert str(error) == "x must be a 2D array"
    else:
        raise AssertionError("一维 x 没有触发 ValueError")

    invalid_w = np.array([1.0, 2.0])
    try:
        linear_forward(x, invalid_w, b)
    except ValueError as error:
        assert str(error) == "w must be a 2D array"
    else:
        raise AssertionError("一维 w 没有触发 ValueError")

    invalid_b = np.array([[1.0, -1.0, 0.0]])
    try:
        linear_forward(x, w, invalid_b)
    except ValueError as error:
        assert str(error) == "b must be a 1D array"
    else:
        raise AssertionError("二维 b 没有触发 ValueError")

    print("linear forward tests passed")

if __name__ == "__main__":
    main()
