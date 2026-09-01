#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>


// 一张图片的一个分类候选结果。
struct ClassPrediction {
    std::size_t class_index;
    std::string class_name;
    float logit;
    float probability;
};


// 对形状为[batch, class_count]的logits执行稳定Softmax，
// 并返回每个样本概率最高的k个类别。
std::vector<std::vector<ClassPrediction>>
computeTopKPredictions(
    const std::vector<float>& logits,
    const std::vector<std::int64_t>& logits_shape,
    const std::vector<std::string>& class_names,
    std::size_t k
);
