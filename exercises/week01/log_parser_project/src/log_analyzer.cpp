#include "log_analyzer.hpp"

#include <algorithm>
#include <set>
#include <tuple>
#include <unordered_map>

namespace {

using LogEntryKey = std::tuple<std::string, std::string, std::string, int>;

LogEntryKey makeLogEntryKey(const LogEntry& entry) {
    return {
        entry.timestamp,
        entry.level,
        entry.path,
        entry.statusCode
    };
}

bool pathCountComesBefore(
    const PathCount& left,
    const PathCount& right
) {
    if (left.second != right.second) {
        return left.second > right.second;
    }

    return left.first < right.first;
}

}  // namespace

LogAnalysis analyzeLogs(
    const LogFileResult& fileResult,
    std::size_t k
) {
    LogAnalysis analysis{};
    std::unordered_map<std::string, std::size_t> pathCounts;
    std::set<LogEntryKey> seenEntries;

    pathCounts.reserve(fileResult.entries.size());

    analysis.validLineCount = fileResult.entries.size();
    analysis.invalidLineCount = fileResult.invalidLineCount;

    for (const LogEntry& entry : fileResult.entries) {
        ++analysis.levelCounts[entry.level];
        ++pathCounts[entry.path];

        if (!seenEntries.insert(makeLogEntryKey(entry)).second) {
            ++analysis.duplicateLineCount;
        }
    }

    analysis.topPaths.reserve(pathCounts.size());

    for (const auto& [path, count] : pathCounts) {
        analysis.topPaths.push_back({path, count});
    }

    std::sort(
        analysis.topPaths.begin(),
        analysis.topPaths.end(),
        pathCountComesBefore
    );

    if (analysis.topPaths.size() > k) {
        analysis.topPaths.resize(k);
    }

    return analysis;
}
