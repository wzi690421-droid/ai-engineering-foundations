# Week 01 Day 02 学习复盘

## 完成内容

- 用 `unordered_map<string, size_t>` 统计词频。
- 把键值对转换成 `vector<pair<string, size_t>>`。
- 按“次数降序、同频单词升序”排序。
- 使用 `resize(k)` 返回前 K 高频词。
- 分别测试计数、排序、同频排序和 `k` 的边界。

## 核心理解

- `++counts[word]`：先通过 `counts[word]` 找到计数；不存在时创建为 `0`，再加一。
- `[word, count]`：拆开已有的 `pair`；`{word, count}`：构造新的 `pair`。
- `unordered_map` 只验证键值关系，不能验证顺序；排序必须使用 `vector` 断言。
- `pair.first` 是整个单词，`pair.second` 是次数。
- 平均时间复杂度为 `O(n + m log m)`，额外空间复杂度为 `O(m)`；`n` 是总单词数，`m` 是不同单词数。

## 当前掌握情况

能够解释整体流程和比较规则，代码与测试运行通过。结构化绑定、`operator[]` 的执行过程和复杂度仍需在后续回忆与闭卷实现中巩固；本次是在提示下完成，尚不算完全独立实现。
