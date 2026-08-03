# Week 01 Day 06 学习复盘

## 完成内容

- 将日志解析、文件读取、等级统计、Top K 路径和重复检测集成为一个工程。
- 新增 `LogAnalysis` 和 `analyzeLogs()`，区分解析结果与分析结果。
- 新增命令行入口，校验文件路径与非负整数 `top-k`。
- 新增固定样例、分析器边界测试和命令行冒烟测试。
- 完成严格编译、3 项 CTest、错误输入和 Sanitizer 验证。

## 数据流与职责

```text
文件路径
  -> loadLogFile()
  -> LogFileResult
  -> analyzeLogs()
  -> LogAnalysis
  -> printAnalysis()
  -> 终端输出
```

- `parseLogLine()`：把一行文本转换成一条 `LogEntry`。
- `loadLogFile()`：打开文件，保存合法日志并统计非法行。
- `analyzeLogs()`：统计等级、完全重复日志和 Top K 路径。
- `main()`：解析参数、连接模块、处理错误并输出结果。

## 关键规则

- 完全重复要求时间戳、等级、路径和状态码四个字段都相同；首次出现不计为重复。
- 路径先按次数降序排列，次数相同时按路径字典序升序排列。
- `k == 0` 返回空 Top K；`k` 大于路径种类时返回全部路径。
- CMake 的配置、构建、测试分别使用 `cmake -S . -B build`、`cmake --build build` 和 `ctest --test-dir build --output-on-failure`。

## 验证结果

- 严格构建通过：`-Wall -Wextra -Werror`。
- 解析器测试、分析器测试、命令行冒烟测试共 3 项全部通过。
- AddressSanitizer 与 UndefinedBehaviorSanitizer 运行无报错。
- 非法 `top-k` 返回 2，文件打不开返回 1，正常执行返回 0。

## 当前掌握情况

能够在引导下说明 `LogEntry -> LogFileResult -> LogAnalysis` 的数据流，并完成接口、基础统计、基础测试和部分 CMake 配置。

Top K 的完整接入、重复检测、命令行入口、扩展测试和文档由 AI 在明确授权后完成。当前功能交付已经完成，但尚未证明能从空文件独立实现同类模块，Day 7 需要闭卷复测。
