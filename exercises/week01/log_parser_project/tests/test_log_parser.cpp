#include "log_parser.hpp"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

namespace {

void testValidLines() {
    LogEntry infoOutput{};

    assert(parseLogLine(
        "2026-07-31T10:15:00 INFO /login 200",
        infoOutput
    ));
    assert(infoOutput.timestamp == "2026-07-31T10:15:00");
    assert(infoOutput.level == "INFO");
    assert(infoOutput.path == "/login");
    assert(infoOutput.statusCode == 200);

    LogEntry warnOutput{};

    assert(parseLogLine(
        "2026-07-31T10:16:00 WARN /health 404",
        warnOutput
    ));
    assert(warnOutput.level == "WARN");
    assert(warnOutput.statusCode == 404);
}

void testInvalidLines() {
    const std::vector<std::string> invalidLines{
        "2026-07-31T10:15:00 INFO /login",
        "2026-07-31T10:15:00 INFO /login abc",
        "2026-07-31T10:15:00 INFO /login 200abc",
        "2026-07-31T10:15:00 INFO /login 200 extra",
        "2026-07-31T10:15:00 INFO /login 99",
        "2026-07-31T10:15:00 INFO /login 600",
        "2026-07-31T10:15:00 NOTICE /login 200",
        "2026-07-31T10:15:00 INFO login 200",
    };

    for (const auto& line : invalidLines) {
        LogEntry output{};
        assert(!parseLogLine(line, output));
    }
}

void testFileLoading() {
    const auto result = loadLogFile(
        "../../../tests/week01/sample_logs.txt"
    );

    assert(result.opened);
    assert(result.entries.size() == 3);
    assert(result.invalidLineCount == 2);
    assert(result.entries[0].path == "/login");
    assert(result.entries[1].statusCode == 503);
    assert(result.entries[2].level == "ERROR");

    const auto missingResult = loadLogFile(
        "../../../tests/week01/file_does_not_exist.txt"
    );

    assert(!missingResult.opened);
    assert(missingResult.entries.empty());
    assert(missingResult.invalidLineCount == 0);
}

}  // namespace

int main() {
    testValidLines();
    testInvalidLines();
    testFileLoading();

    std::cout << "All tests passed\n";
    return 0;
}
