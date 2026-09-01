#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>


// 保存训练阶段使用的三个通道归一化参数。
//
// std::array<float, 3>表示长度固定为3，
// 分别对应RGB三个颜色通道。
struct NormalizationParameters {
    std::array<float, 3> mean;
    std::array<float, 3> std;
};


// 保存一批图片经过预处理后的Tensor数据。
//
// 例如两张图片：
// shape  = [2, 3, 32, 32]
// values = 6144个连续float
struct ImageBatch {
    std::vector<std::int64_t> shape;
    std::vector<float> values;
};


// 从训练产生的config.json中读取mean和std。
//
// 使用配置文件作为唯一数据来源，避免C++和Python
// 分别手写两套不同的归一化参数。
NormalizationParameters loadNormalizationParameters(
    const std::string& config_path
);


// 负责把一张或多张图片转换为模型需要的NCHW Tensor。
class ImagePreprocessor {
public:
    // 创建预处理器时固定：
    ImagePreprocessor(
        int target_height,
        int target_width,
        NormalizationParameters normalization
    );

    // 接收一张或多张图片路径。
    ImageBatch preprocess(
        const std::vector<std::string>& image_paths
    ) const;

private:
    int target_height_;
    int target_width_;

    // 长期保存训练阶段使用的mean和std。
    NormalizationParameters normalization_;
};
