#include "image_preprocessor.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


NormalizationParameters loadNormalizationParameters(
    const std::string& config_path
) {
    // 按JSON格式，以只读方式打开训练配置。
    cv::FileStorage config_file{
        config_path,
        cv::FileStorage::READ |
        cv::FileStorage::FORMAT_JSON
    };

    if (!config_file.isOpened()) {
        throw std::runtime_error(
            "failed to open config file: " +
            config_path
        );
    }

    // 读取JSON中的normalization对象。
    const cv::FileNode normalization_node =
        config_file["normalization"];

    if ( normalization_node.empty() ||
         !normalization_node.isMap()
    ) {
        throw std::runtime_error(
            "config does not contain a valid "
            "normalization object"
        );
    }

    // 读取normalization中的两个数组。
    const cv::FileNode mean_node = normalization_node["mean"];

    const cv::FileNode std_node = normalization_node["std"];

    // 创建准备返回的参数，六个float先初始化为0。
    NormalizationParameters parameters{};

    // mean和std必须都是长度为3的数组。
    if (
        !mean_node.isSeq() ||
        !std_node.isSeq() ||
        mean_node.size() != parameters.mean.size() ||
        std_node.size() != parameters.std.size()
    ) {
        throw std::runtime_error(
            "normalization mean and std must "
            "each contain three values"
        );
    }

    // 逐个读取RGB三个通道。
    for (
        std::size_t channel = 0;
        channel < parameters.mean.size();
        ++channel
    ) {
        const cv::FileNode mean_value_node =
            mean_node[static_cast<int>(channel)];

        const cv::FileNode std_value_node =
            std_node[static_cast<int>(channel)];

        // 每个节点都必须是数字。
        if (
            (!mean_value_node.isReal() &&
             !mean_value_node.isInt()) ||
            (!std_value_node.isReal() &&
             !std_value_node.isInt())
        ) {
            throw std::runtime_error(
                "normalization values must be numeric"
            );
        }

        // 把JSON数字保存为float。
        mean_value_node >>
            parameters.mean[channel];

        std_value_node >>
            parameters.std[channel];
    }

    // FileStorage会自动关闭配置文件。
    return parameters;
}


ImagePreprocessor::ImagePreprocessor(
    int target_height,
    int target_width,
    NormalizationParameters normalization
)
    // 把调用者传入的参数保存到成员变量。
    : target_height_{target_height},
      target_width_{target_width},
      normalization_{normalization} {
    // 图片目标尺寸必须是正数。
    //
    // 例如32×32合法；
    // 0×32或-1×32没有意义。
    if (
        target_height_ <= 0 ||
        target_width_ <= 0
    ) {
        throw std::invalid_argument(
            "target image dimensions must be positive"
        );
    }

    // 分别检查RGB三个通道。
    for (
        std::size_t channel = 0;
        channel < normalization_.mean.size();
        ++channel
    ) {
        // mean和std不能是NaN或无穷大。
        if (
            !std::isfinite(normalization_.mean[channel]) ||
            !std::isfinite(normalization_.std[channel])
        ) {
            throw std::invalid_argument(
                "normalization parameters must be finite"
            );
        }

        // 标准化公式是：
        //
        // normalized = (pixel - mean) / std
        //
        // std等于0会发生除以零；
        // std小于0也不符合标准差定义。
        if (normalization_.std[channel] <= 0.0F) {
            throw std::invalid_argument(
                "normalization standard deviations "
                "must be positive"
            );
        }
    }
}


ImageBatch ImagePreprocessor::preprocess(
    // 接收一张或多张图片的路径。
    // const&表示不复制整个路径数组，也不允许修改它。
    const std::vector<std::string>& image_paths
) const {
    // 至少需要一张图片。
    //
    // 否则batch维等于0，无法形成合法的模型输入。
    if (image_paths.empty()) {
        throw std::invalid_argument(
            "at least one image path is required"
        );
    }

    // 创建最终需要返回的数据。
    //
    // 此时shape和values还是两个空vector。
    ImageBatch batch;

    // 设置最终Tensor形状：
    //
    // [N, C, H, W]
    //
    // N：图片数量；
    // C：RGB三个通道；
    // H：目标高度；
    // W：目标宽度。
    batch.shape = {
        static_cast<std::int64_t>(image_paths.size()),
        3,
        static_cast<std::int64_t>(target_height_),
        static_cast<std::int64_t>(target_width_)
    };

    // 每张图片最终需要保存多少个float。
    //
    // 对32×32 RGB图片：
    // 3×32×32=3072。
    const std::size_t values_per_image =
        normalization_.mean.size() *
        static_cast<std::size_t>(target_height_) *
        static_cast<std::size_t>(target_width_);

    // 提前为整个batch申请足够空间。
    //
    // 两张图片时reserve 6144个float。
    //
    // reserve只申请容量，不会改变vector的size。
    // 后面仍然通过push_back逐个加入数据。
    batch.values.reserve(
        image_paths.size() *
        values_per_image
    );

    // 逐张处理图片。
    for (const std::string& image_path : image_paths) {
        // OpenCV默认按照BGR顺序读取。
        //
        // IMREAD_COLOR保证读取结果是三通道彩色图：
        // 灰度图会转成三通道；
        // 带透明度的图片会舍弃Alpha通道。
        cv::Mat bgr_image =
            cv::imread(
                image_path,
                cv::IMREAD_COLOR
            );

        // 图片不存在、损坏或者格式无法识别时，
        // imread不会一定抛出异常，而是返回空Mat。
        if (bgr_image.empty()) {
            throw std::runtime_error(
                "failed to read image: " +
                image_path
            );
        }

        // 创建用于保存RGB结果的Mat。
        cv::Mat rgb_image;

        // OpenCV读取顺序是BGR，
        // 但训练阶段的PIL/ToTensor按照RGB处理。
        //
        // 因此必须交换第0和第2通道：
        // BGR → RGB。
        cv::cvtColor(
            bgr_image,
            rgb_image,
            cv::COLOR_BGR2RGB
        );

        // 创建用于保存缩放结果的Mat。
        cv::Mat resized_image;

        // 把图片统一缩放到模型需要的宽度和高度。
        //
        // 注意cv::Size的参数顺序是：
        // width在前，height在后。
        cv::resize(
            rgb_image,
            resized_image,
            cv::Size(
                target_width_,
                target_height_
            ),
            0.0,
            0.0,
            cv::INTER_LINEAR
        );

        // 创建float32图片。
        cv::Mat float_image;

        // resized_image当前是uint8：
        // 每个通道数值范围为0～255。
        //
        // CV_32FC3表示：
        // 32位float，三个通道。
        //
        // 1.0/255.0是缩放系数：
        // 0～255 → 0～1。
        resized_image.convertTo(
            float_image,
            CV_32FC3,
            1.0 / 255.0
        );

        // OpenCV的Mat内存排列是HWC：
        //
        // pixel(0,0)的R、G、B
        // pixel(0,1)的R、G、B
        // ...
        //
        // 模型需要NCHW，因此先遍历通道。
        for (
            std::size_t channel = 0;
            channel < normalization_.mean.size();
            ++channel
        ) {
            // 当前通道对应的训练均值。
            const float channel_mean =
                normalization_.mean[channel];

            // 当前通道对应的训练标准差。
            const float channel_std =
                normalization_.std[channel];

            // 遍历图片每一行。
            for (
                int row = 0;
                row < target_height_;
                ++row
            ) {
                // 取得当前行第一个RGB像素的地址。
                //
                // cv::Vec3f表示一个像素中有三个float：
                // [R, G, B]。
                const cv::Vec3f* row_pixels =
                    float_image.ptr<cv::Vec3f>(row);

                // 遍历当前行的每一列。
                for (
                    int column = 0;
                    column < target_width_;
                    ++column
                ) {
                    // 取得当前像素的当前颜色通道。
                    //
                    // row_pixels[column]表示当前像素；
                    // [channel]取得其中的R、G或B。
                    const float pixel_value =
                        row_pixels[column][
                            static_cast<int>(channel)
                        ];

                    // 使用训练阶段的相同公式进行归一化：
                    //
                    // normalized =
                    //     (pixel - mean) / std
                    const float normalized_value =
                        ( pixel_value - channel_mean) / channel_std;

                    // 按照NCHW顺序依次加入vector：
                    //
                    // 第一张图片全部R
                    // → 第一张图片全部G
                    // → 第一张图片全部B
                    // → 第二张图片全部R
                    // → ...
                    batch.values.push_back(
                        normalized_value
                    );
                }
            }
        }
    }

    // 检查最终写入的元素数量。
    //
    // 如果这里不相等，说明上面的循环或形状计算存在错误。
    const std::size_t expected_value_count =
        image_paths.size() * values_per_image;

    if (batch.values.size() != expected_value_count) {
        throw std::runtime_error(
            "preprocessed image buffer has "
            "an unexpected size"
        );
    }

    // batch拥有自己的shape和连续float数据。
    return batch;
}
