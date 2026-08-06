import numpy as np

from model import (
    cross_entropy_from_logits,
    two_layer_backward,
    two_layer_forward,
)


def make_xor_data(samples_per_cluster=100, noise=0.25, seed=42):
    rng = np.random.default_rng(seed)

    centers = np.array([
        [-1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
    ])

    center_labels = np.array([0, 0, 1, 1])

    inputs = []
    labels = []

    for center, label in zip(centers, center_labels):
        noise_values = rng.normal(
            loc=0.0,
            scale=noise,
            size=(samples_per_cluster, 2),
        )
        cluster = center + noise_values

        cluster_labels = np.full(
            samples_per_cluster,
            label,
            dtype=np.int64,
        )

        inputs.append(cluster)
        labels.append(cluster_labels)

    x = np.vstack(inputs)
    y = np.concatenate(labels)
    order = rng.permutation(y.shape[0])
    return x[order], y[order]


def initialize_parameters(input_size, hidden_size, class_count, seed=7):
    rng = np.random.default_rng(seed)

    w1 = rng.standard_normal(
        (input_size, hidden_size)
    ) * np.sqrt(2.0 / input_size)

    b1 = np.zeros(hidden_size)

    w2 = rng.standard_normal(
        (hidden_size, class_count)
    ) * np.sqrt(2.0 / hidden_size)

    b2 = np.zeros(class_count)

    parameters = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
    }

    return parameters


def evaluate(x, labels, parameters):
    probabilities, cache = two_layer_forward(
        x,
        parameters["w1"],
        parameters["b1"],
        parameters["w2"],
        parameters["b2"],
    )

    loss = cross_entropy_from_logits(
        cache["logits"],
        labels,
    )

    predictions = np.argmax(probabilities, axis=1)

    accuracy = np.mean(predictions == labels)

    return loss, accuracy


def train_step(x, labels, parameters, learning_rate):
    probabilities, cache = two_layer_forward(
        x,
        parameters["w1"],
        parameters["b1"],
        parameters["w2"],
        parameters["b2"],
    )

    loss = cross_entropy_from_logits(
        cache["logits"],
        labels,
    )

    predictions = np.argmax(probabilities, axis=1)

    accuracy = np.mean(predictions == labels)

    gradients = two_layer_backward(
        labels,
        cache,
        parameters["w2"],
    )

    parameters["w1"] -= learning_rate * gradients["dw1"]
    parameters["b1"] -= learning_rate * gradients["db1"]
    parameters["w2"] -= learning_rate * gradients["dw2"]
    parameters["b2"] -= learning_rate * gradients["db2"]

    return loss, accuracy


def main():
    train_x, train_labels = make_xor_data(seed=42)

    test_x, test_labels = make_xor_data(seed=2026)

    learning_rates = [0.001, 0.1, 10.0]

    total_steps = 500
    record_interval = 100

    results = []

    for learning_rate in learning_rates:
        parameters = initialize_parameters(
            input_size=2,
            hidden_size=8,
            class_count=2,
            seed=7,
        )

        print(f"\nlearning_rate={learning_rate}")

        for step in range(1, total_steps + 1):
            train_step(
                train_x,
                train_labels,
                parameters,
                learning_rate=learning_rate,
            )

            if step % record_interval == 0:
                loss, accuracy = evaluate(
                    train_x,
                    train_labels,
                    parameters,
                )

                test_loss, test_accuracy = evaluate(
                    test_x,
                    test_labels,
                    parameters,
                )

                results.append([
                    learning_rate,
                    step,
                    loss,
                    accuracy,
                    test_loss,
                    test_accuracy,
                ])

                print(
                    f"step={step:3d} "
                    f"train_loss={loss:.6f} "
                    f"train_accuracy={accuracy:.3f} "
                    f"test_loss={test_loss:.6f} "
                    f"test_accuracy={test_accuracy:.3f}"
                )

    results_array = np.array(results)

    np.savetxt(
        "learning_rate_results.csv",
        results_array,
        delimiter=",",
        header=(
            "learning_rate,step,train_loss,train_accuracy,"
            "test_loss,test_accuracy"
        ),
        comments="",
    )

    print("\nResults saved to learning_rate_results.csv")


if __name__ == "__main__":
    main()
