#pragma once

#include "log_parser.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using PathCount = std::pair<std::string, std::size_t>;

struct LogAnalysis {
    std::size_t validLineCount{};
    std::size_t invalidLineCount{};
    std::size_t duplicateLineCount{};
    std::unordered_map<std::string, std::size_t> levelCounts;
    std::vector<PathCount> topPaths;
};

LogAnalysis analyzeLogs(
    const LogFileResult& fileResult,
    std::size_t k
);
