#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct LogEntry {
    std::string timestamp;
    std::string level;
    std::string path;
    int statusCode;
};

struct LogFileResult {
    std::vector<LogEntry> entries;
    std::size_t invalidLineCount;
    bool opened;
};

bool parseLogLine(
    const std::string& line,
    LogEntry& output
);

LogFileResult loadLogFile(
    const std::string& filePath
);