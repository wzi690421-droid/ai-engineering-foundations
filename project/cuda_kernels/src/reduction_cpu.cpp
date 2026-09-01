#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

// 串行参考实现：从头到尾依次累加所有元素。
// 后续多线程和 CUDA 版本都必须与这个函数的结果一致。
std::int64_t sumSerial(const std::vector<int>& values) {
    // 使用 int64_t 保存总和，降低大量 int 相加时溢出的风险。
    std::int64_t sum = 0;

    // const int& 表示只读取 vector 中的原始元素，不复制、不修改。
    for (const int& value : values) {
        sum += value;
    }

    return sum;
}

int main() {
    const std::vector<int> values{3, 1, 4, 1, 5};

    const std::int64_t result = sumSerial(values);

    // 这是当前阶段的最小正确性检查，不是正式测试框架。
    assert(result == 14);
    std::cout << "serial sum: " << result << '\n';

    return 0;
}
