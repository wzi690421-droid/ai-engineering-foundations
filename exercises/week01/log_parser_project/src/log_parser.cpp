#include "log_parser.hpp"

#include <fstream>
#include <sstream>

namespace {

bool isValidLevel(const std::string& level) {
    return level == "DEBUG"
        || level == "INFO"
        || level == "WARN"
        || level == "ERROR";
}

}  // namespace

bool parseLogLine(
    const std::string& line,
    LogEntry& output
) {
    std::istringstream stream(line);
    LogEntry parsed{};

    if (!(stream >> parsed.timestamp
                 >> parsed.level
                 >> parsed.path
                 >> parsed.statusCode)) {
        return false;
    }

    std::string extra;

    if (stream >> extra) {
        return false;
    }

    if (parsed.statusCode < 100 || parsed.statusCode > 599) {
        return false;
    }

    if (!isValidLevel(parsed.level)) {
        return false;
    }

    if (parsed.path.front() != '/') {
        return false;
    }

    output = parsed;
    return true;
}

LogFileResult loadLogFile(const std::string& filePath) {
    LogFileResult result{};
    std::ifstream file(filePath);

    if (!file.is_open()) {
        return result;
    }

    result.opened = true;

    std::string line;

    while (std::getline(file, line)) {
        LogEntry entry{};

        if (parseLogLine(line, entry)) {
            result.entries.push_back(entry);
        } else {
            ++result.invalidLineCount;
        }
    }

    return result;
}
