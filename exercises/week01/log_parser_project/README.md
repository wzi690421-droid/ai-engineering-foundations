# Log Analyzer v0.1

一个 C++17 命令行日志分析器。程序读取日志文件，过滤非法行，并输出等级统计、完全重复日志数量和访问次数最多的前 K 个路径。

## 日志格式

每一行包含四个以空格分隔的字段：

```text
timestamp level path status-code
```

示例：

```text
2026-08-03T09:00:00 INFO /login 200
```

等级只接受 `DEBUG`、`INFO`、`WARN` 和 `ERROR`；路径必须以 `/` 开头；状态码范围是 100 到 599；多余或缺少字段的行视为非法。

完全重复表示 `timestamp`、`level`、`path` 和 `statusCode` 四个字段全部相同。首次出现不计入重复数量，之后每次重复各计一次。

## 数据流

```text
文件路径
  -> loadLogFile()
  -> LogFileResult
  -> analyzeLogs()
  -> LogAnalysis
  -> printAnalysis()
  -> 终端输出
```

## 构建与测试

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## 运行

```bash
./build/log_analyzer tests/data/sample.log 3
```

输出：

```text
valid lines: 5
invalid lines: 1
duplicate lines: 1

levels:
DEBUG: 0
INFO: 3
WARN: 1
ERROR: 1

top paths:
1. /login 3
2. /health 1
3. /orders 1
```

## 文件职责

- `include/log_parser.hpp`、`src/log_parser.cpp`：解析单行并读取文件。
- `include/log_analyzer.hpp`、`src/log_analyzer.cpp`：统计等级、重复日志和 Top K 路径。
- `app/main.cpp`：处理命令行参数并输出结果。
- `tests/`：解析器测试、分析器测试和固定样例数据。
