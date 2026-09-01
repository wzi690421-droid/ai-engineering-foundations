# 阶段 01：C++ 工程基础

## 这一阶段解决了什么

目标不是刷语法，而是第一次走通“小需求 → 数据结构 → 实现 → 测试 → 构建 → 调试 → 交付”。最终成果是一个 C++17 日志分析器。

```text
日志文件
→ parseLogLine() 解析一行
→ loadLogFile() 读取全部行
→ analyzeLogs() 统计等级、重复项和 Top K
→ printAnalysis() 输出结果
```

## 容器、引用与数据结构

- `const std::vector<int>& nums`：只读借用外部数组，避免复制，也禁止修改。
- `LogEntry& output`：可修改引用，函数成功后可把结果写回调用者。
- 解析时先写临时对象 `parsed`，全部验证通过后再赋给 `output`，避免失败时留下半成品。
- `std::pair` 适合简单二元关系；字段多、含义复杂时用 `struct`。
- `unordered_map<Key, Value>` 保存键值关系，不保证遍历顺序；需要排序时先转成 `vector<pair<...>>`。
- `++counts[word]`：不存在时 `operator[]` 先创建值 `0`，再执行加一。
- `[word, count]` 是拆开已有 `pair`；`{word, count}` 是构造一个新对象。

## 排序、去重与 Top K

```text
sort(greater)
→ 相同元素相邻且整体降序
→ unique 把重复项移动到有效区间之后
→ erase 真正缩短容器
→ resize(k) 截取前 K 项
```

`unique` 不会删除元素，也不会自动排序，只返回“新逻辑结尾”的迭代器。`find` 返回的也是迭代器；查找失败用 `iterator == container.end()` 判断。

词频和路径 Top K 的比较规则是：次数降序；次数相同时，字符串字典序升序。

设输入数量为 `n`，不同键数量为 `m`：

```text
哈希统计：平均 O(n)
转成数组：O(m)
排序：O(m log m)
总时间：平均 O(n + m log m)
额外空间：O(m)
```

双重循环检测重复最坏为 `O(n²)`、额外空间 `O(1)`；哈希集合平均 `O(n)`、额外空间 `O(n)`。数据很少且重复很早时，简单循环可能更快；不能脱离输入分布只背复杂度。

## 文本解析与边界验证

日志格式：

```text
timestamp level path status-code
```

解析顺序：

1. 用 `istringstream` 读取四个字段。
2. `stream >> extra` 如果还能读出内容，说明存在多余字段。
3. 验证等级集合、路径首字符和状态码范围。
4. 全部成功后写入输出对象。

`ifstream file(path)` 才负责打开文件；`result.opened` 只是记录打开结果的布尔值。

## CMake 工程关系

```text
include/*.hpp：声明公共类型和函数接口
src/*.cpp：实现函数逻辑
app/main.cpp：命令行入口
tests/*.cpp：测试入口
```

- `#include "log_parser.hpp"` 发出头文件查找请求。
- `target_include_directories()` 提供搜索路径。
- `add_library()` 生成可链接库；`add_executable()` 生成含 `main()` 的程序。
- 头文件目录使用 `PUBLIC`，依赖该库的目标也需要它；严格警告使用 `PRIVATE`，只约束当前目标。

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

三条命令分别是配置生成、编译链接、运行测试，不是同一件事。

## 调试工具与关键教训

- `-Wall -Wextra -Werror`：把潜在问题尽早暴露；`-Werror` 会把警告升级为编译错误。
- 编译失败不会删除旧可执行文件。使用 `g++ ... -o app && ./app`，避免误运行旧版本。
- ASan 的 `heap-buffer-overflow` 表示越界；`READ of size 4` 表示本次读取 4 字节，不表示越界距离是 4 字节。
- 数组下标从 0 开始，长度 3 的合法下标是 `0、1、2`；条件应为 `i < size()`。
- GDB：`break` 设置断点，`run` 启动，`next` 执行当前行，`continue` 到下一断点，`finish` 运行到当前函数返回。
- `double result = 5 / 2` 得到 `2.0`，因为整数除法先发生；应让至少一个操作数先变成浮点数。

## Git 基础

- `git add` 把修改放入暂存区，`git commit` 生成本地提交；提交不等于上传 GitHub。
- 路径相对于当前终端目录。已经位于 `exercises/week01` 时，不应再次写 `exercises/week01/file.cpp`。
- 查看指定文件：`git diff -- path`；查看暂存内容：`git diff --cached -- path`。

## 已完成证据与薄弱点

- 日志解析、分析、命令行和 3 项 CTest 全部通过。
- 严格编译、AddressSanitizer 和 UndefinedBehaviorSanitizer 通过。
- 闭卷需求变更完成，阶段评分 `3/5`。
- 仍需巩固：从空目录独立搭 CMake、多文件接口设计、复杂度完整拆分，以及减少代码补全依赖。

## 复盘自测

1. 为什么 `unique` 后还需要 `erase`？
2. 为什么哈希统计后还要转成数组排序？
3. 为什么解析失败前不能直接修改 `output`？
4. `PUBLIC` 头文件路径与 `PRIVATE` 编译选项分别影响谁？
5. 编译失败后 `./app` 为什么仍可能运行？
