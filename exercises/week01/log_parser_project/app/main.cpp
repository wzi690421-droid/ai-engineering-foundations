#include "log_analyzer.hpp"
#include "log_parser.hpp"

#include <array>
#include <charconv>
#include <cstddef>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>

namespace {

bool parseTopK(std::string_view text, std::size_t& output) {
    if (text.empty()) {
        return false;
    }

    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto [position, error] = std::from_chars(begin, end, output);

    return error == std::errc{} && position == end;
}

void printAnalysis(const LogAnalysis& analysis) {
    std::cout << "valid lines: " << analysis.validLineCount << '\n';
    std::cout << "invalid lines: " << analysis.invalidLineCount << '\n';
    std::cout << "duplicate lines: " << analysis.duplicateLineCount << "\n\n";

    std::cout << "levels:\n";

    constexpr std::array<std::string_view, 4> levels{
        "DEBUG", "INFO", "WARN", "ERROR"
    };

    for (const std::string_view level : levels) {
        const auto found = analysis.levelCounts.find(std::string(level));
        const std::size_t count = found == analysis.levelCounts.end()
            ? 0
            : found->second;

        std::cout << level << ": " << count << '\n';
    }

    std::cout << "\ntop paths:\n";

    for (std::size_t index = 0; index < analysis.topPaths.size(); ++index) {
        const auto& [path, count] = analysis.topPaths[index];
        std::cout << index + 1 << ". " << path << ' ' << count << '\n';
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <log-file> <top-k>\n";
        return 2;
    }

    std::size_t k{};

    if (!parseTopK(argv[2], k)) {
        std::cerr << "top-k must be a non-negative integer\n";
        return 2;
    }

    const LogFileResult fileResult = loadLogFile(argv[1]);

    if (!fileResult.opened) {
        std::cerr << "failed to open log file: " << argv[1] << '\n';
        return 1;
    }

    const LogAnalysis analysis = analyzeLogs(fileResult, k);
    printAnalysis(analysis);
    return 0;
}
