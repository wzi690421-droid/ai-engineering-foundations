# Day 07 闭卷周测：请求统计模块

## 规则

- 第一部分限时 60 分钟。
- 不查看 Week 01 旧代码，不让 AI 提供或修改核心代码。
- 可以使用编译器、GDB 和 AddressSanitizer。
- 在 `solution.cpp` 中从空文件完成类型、函数、测试和 `main()`。
- 如需帮助，记录帮助内容；提示等级会计入独立性评分。

## 第一部分：从空文件实现

定义以下数据类型：

```cpp
struct Request {
    std::string path;
    int statusCode;
};

using PathCount = std::pair<std::string, std::size_t>;

struct RequestSummary {
    std::size_t totalCount;
    std::size_t serverErrorCount;
    std::vector<PathCount> topPaths;
};
```

实现函数：

```cpp
RequestSummary summarizeRequests(
    const std::vector<Request>& requests,
    std::size_t k
);
```

功能要求：

1. `totalCount` 等于全部请求数量。
2. 状态码在 500 到 599 之间时计入 `serverErrorCount`。
3. 统计每个路径在全部请求中出现的次数。
4. `topPaths` 按次数从大到小排列。
5. 次数相同时，路径按字典序从小到大排列。
6. 最多返回前 `k` 个路径。
7. `k == 0` 时 `topPaths` 为空。
8. `k` 大于不同路径数量时返回全部路径。
9. 不修改输入数组。

## 必须自行编写的测试

至少覆盖：

1. 普通混合输入；
2. 空输入；
3. `k == 0`；
4. `k` 大于不同路径数量；
5. 两个路径次数相同，验证字典序；
6. 状态码 499、500、599、600 的边界。

## 严格编译

```bash
g++ -std=c++17 -Wall -Wextra -Werror solution.cpp -o solution
./solution
```

测试全部通过时输出：

```text
Day 7 part 1 passed
```

## 提交前说明

完成后不要立即修改。记录：

- 实际用时；
- 使用过的编译或调试命令；
- 遇到的第一个错误及其根本原因；
- 时间复杂度和空间复杂度；
- 是否查看旧代码或获得提示。

随后进入第二部分：现场需求变更。

## 周测结果

- 第一部分在 60 分钟内完成核心函数和测试框架，未查看旧代码；使用了 VS Code 内部代码提示。
- 初次提交因 `assert` 宏、测试期望次数错误而未通过，经 AI 诊断后修正。
- 第二部分在规定时间内完成：Top K 改为只统计状态码 200～399 的成功请求。
- 严格编译、普通运行、AddressSanitizer 和 UndefinedBehaviorSanitizer 均通过。
- 空间复杂度回答为 `O(m)`，正确；时间复杂度漏掉排序，应为平均 `O(n + m log m)`。
- Week 01 最终评分：`3/5`，达到进入 Week 02 的标准。
