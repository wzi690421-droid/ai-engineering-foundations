#include "classification_postprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>


std::vector<std::vector<ClassPrediction>>
computeTopKPredictions(
    const std::vector<float>& logits,
    const std::vector<std::int64_t>& logits_shape,
    const std::vector<std::string>& class_names,
    std::size_t k
) {
    if (
        logits_shape.size() != 2 ||
        logits_shape[0] <= 0 ||
        logits_shape[1] <= 0
    ) {
        throw std::invalid_argument(
            "logits shape must be [positive batch, positive class count]"
        );
    }

    const std::size_t batch_size =
        static_cast<std::size_t>(logits_shape[0]);

    const std::size_t class_count =
        static_cast<std::size_t>(logits_shape[1]);

    if (class_names.size() != class_count) {
        throw std::invalid_argument(
            "class name count does not match logits class count"
        );
    }

    if (k == 0 || k > class_count) {
        throw std::invalid_argument(
            "top-k must be between 1 and the class count"
        );
    }

    if (
        batch_size >
        std::numeric_limits<std::size_t>::max() / class_count ||
        logits.size() != batch_size * class_count
    ) {
        throw std::invalid_argument(
            "logit buffer size does not match logits shape"
        );
    }

    std::vector<std::vector<ClassPrediction>> batch_predictions;
    batch_predictions.reserve(batch_size);

    for (
        std::size_t sample_index = 0;
        sample_index < batch_size;
        ++sample_index
    ) {
        const float* sample_logits =
            logits.data() + sample_index * class_count;

        for (
            std::size_t class_index = 0;
            class_index < class_count;
            ++class_index
        ) {
            if (!std::isfinite(sample_logits[class_index])) {
                throw std::invalid_argument(
                    "logits must contain only finite values"
                );
            }
        }

        const float maximum_logit =
            *std::max_element(
                sample_logits,
                sample_logits + class_count
            );

        std::vector<double> probabilities(class_count);
        double exponential_sum = 0.0;

        for (
            std::size_t class_index = 0;
            class_index < class_count;
            ++class_index
        ) {
            const double exponential = std::exp(
                static_cast<double>(sample_logits[class_index]) -
                static_cast<double>(maximum_logit)
            );

            probabilities[class_index] = exponential;
            exponential_sum += exponential;
        }

        if (
            !std::isfinite(exponential_sum) ||
            exponential_sum <= 0.0
        ) {
            throw std::runtime_error(
                "softmax produced an invalid exponential sum"
            );
        }

        for (double& probability : probabilities) {
            probability /= exponential_sum;
        }

        std::vector<std::size_t> sorted_indices(class_count);
        std::iota(
            sorted_indices.begin(),
            sorted_indices.end(),
            0
        );

        std::partial_sort(
            sorted_indices.begin(),
            sorted_indices.begin() + static_cast<std::ptrdiff_t>(k),
            sorted_indices.end(),
            [&probabilities](
                std::size_t left,
                std::size_t right
            ) {
                if (probabilities[left] != probabilities[right]) {
                    return probabilities[left] > probabilities[right];
                }

                return left < right;
            }
        );

        std::vector<ClassPrediction> sample_predictions;
        sample_predictions.reserve(k);

        for (
            std::size_t rank = 0;
            rank < k;
            ++rank
        ) {
            const std::size_t class_index = sorted_indices[rank];

            sample_predictions.push_back(
                ClassPrediction{
                    class_index,
                    class_names[class_index],
                    sample_logits[class_index],
                    static_cast<float>(probabilities[class_index])
                }
            );
        }

        batch_predictions.push_back(
            std::move(sample_predictions)
        );
    }

    return batch_predictions;
}
