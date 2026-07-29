# Week 01

## 任务 1：返回不重复的前 K 大元素

在 `top_k.cpp` 中独立实现：

```cpp
std::vector<int> topKDistinct(
    const std::vector<int>& nums,
    std::size_t k
);
```

要求：

- 返回不重复的前 `k` 大元素；
- 结果按从大到小排列；
- 不修改输入；
- 正确处理空数组、重复元素、负数和 `k = 0`；
- 自己编写 `main()` 和边界测试；
- 使用 C++17 编译；
- 暂时不要让 AI 生成完整答案。

编译检查：

```bash
g++ -std=c++17 -Wall -Wextra -Werror top_k.cpp -o top_k
./top_k
```

完成后记录：

- 最终代码；
- 时间复杂度和空间复杂度；
- 遇到的编译错误；
- 自己尝试过但失败的方法；
- AI 是否参与以及参与方式。

