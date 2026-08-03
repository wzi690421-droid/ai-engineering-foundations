#include "log_analyzer.hpp"

#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

void testCountsTopPathsAndDuplicates() {
    LogFileResult input{};

    input.opened = true;
    input.invalidLineCount = 2;
    input.entries = {
        {"t1", "INFO", "/login", 200},
        {"t1", "INFO", "/login", 200},
        {"t2", "ERROR", "/login", 500},
        {"t3", "WARN", "/health", 503}
    };

    const LogAnalysis result = analyzeLogs(input, 2);

    assert(result.validLineCount == 4);
    assert(result.invalidLineCount == 2);
    assert(result.levelCounts.at("INFO") == 2);
    assert(result.levelCounts.at("ERROR") == 1);
    assert(result.levelCounts.at("WARN") == 1);
    assert(result.duplicateLineCount == 1);
    assert((result.topPaths == std::vector<PathCount>{
        {"/login", 3},
        {"/health", 1}
    }));

    const LogAnalysis noTopPaths = analyzeLogs(input, 0);
    assert(noTopPaths.topPaths.empty());
    assert(noTopPaths.duplicateLineCount == 1);
}

void testTieBreakAndLargeK() {
    LogFileResult input{};
    input.opened = true;
    input.entries = {
        {"t1", "INFO", "/z", 200},
        {"t2", "INFO", "/a", 200}
    };

    const LogAnalysis result = analyzeLogs(input, 10);

    assert((result.topPaths == std::vector<PathCount>{
        {"/a", 1},
        {"/z", 1}
    }));
    assert(result.duplicateLineCount == 0);
}

void testEmptyEntries() {
    LogFileResult input{};
    input.opened = true;
    input.invalidLineCount = 3;

    const LogAnalysis result = analyzeLogs(input, 5);

    assert(result.validLineCount == 0);
    assert(result.invalidLineCount == 3);
    assert(result.duplicateLineCount == 0);
    assert(result.levelCounts.empty());
    assert(result.topPaths.empty());
}

}  // namespace

int main() {
    testCountsTopPathsAndDuplicates();
    testTieBreakAndLargeK();
    testEmptyEntries();

    std::cout << "All analyzer tests passed\n";
    return 0;
}
