#include <string>
#include <sstream>
#include <cassert>
#include <iostream>
#include <cstddef>
#include <fstream>
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

bool isValidLevel(const std::string& level) {
    return level == "DEBUG"
        || level == "INFO"
        || level == "WARN"
        || level == "ERROR";
}

bool parseLogLine(
    const std::string& line,
    LogEntry& output
){
    std::istringstream stream(line);
    LogEntry parsed{};

    if(!(stream >> parsed.timestamp
                >> parsed.level
                >> parsed.path
                >> parsed.statusCode)){
        return false;
    }
    
    std::string extra;

    if(stream >> extra){
        return false;
    }

    if(parsed.statusCode < 100 || parsed.statusCode > 599){
        return false;
    }

    if(!isValidLevel(parsed.level)){
        return false;
    }

    if(parsed.path.front() != '/'){
        return false;
    }

    output = parsed;
    return true;
}

LogFileResult loadLogFile(const std::string& filePath){
    LogFileResult result{};

    std::ifstream file(filePath);

    if(!file.is_open()){
        return result;
    }

    result.opened = true;

    std::string line;
    
    while(std::getline(file,line)){
        LogEntry entry{};

        if(parseLogLine(line,entry)){
            result.entries.push_back(entry);
        }else{
            ++result.invalidLineCount;
        }
    }
    

    return result;
}

int main() {
    // 测试1：正确日志
    LogEntry validOutput{};

    bool validResult = parseLogLine(
        "2026-07-31T10:15:00 INFO /login 200",
        validOutput
    );

    assert(validResult == true);
    assert(validOutput.timestamp == "2026-07-31T10:15:00");
    assert(validOutput.level == "INFO");
    assert(validOutput.path == "/login");
    assert(validOutput.statusCode == 200);


    // 测试2：缺少状态码
    LogEntry missingStatusOutput{};

    bool missingStatusResult = parseLogLine(
        "2026-07-31T10:15:00 INFO /login",
        missingStatusOutput
    );

    assert(missingStatusResult == false);


    // 测试3：状态码不是整数
    LogEntry invalidStatusOutput{};

    bool invalidStatusResult = parseLogLine(
        "2026-07-31T10:15:00 INFO /login abc",
        invalidStatusOutput
    );

    assert(invalidStatusResult == false);

    // 测试4：状态码后面粘有字母
    LogEntry partialNumberOutput{};

    bool partialNumberResult = parseLogLine(
        "2026-07-31T10:15:00 INFO /login 200abc",
        partialNumberOutput
    );

    assert(partialNumberResult == false);
    
    // 测试5：正确字段后还有多余内容
    LogEntry extraFieldOutput{};

    bool extraFieldResult = parseLogLine(
        "2026-07-31T10:15:00 INFO /login 200 extra",
        extraFieldOutput
    );
    
    assert(extraFieldResult == false);

    // 测试6：是否小于合法状态码
    LogEntry tooSmallStatusOutput{};

    assert(parseLogLine(
        "2026-07-31T10:15:00 INFO /login 99",
        tooSmallStatusOutput
    ) == false);
     
    //测试7：是否大于合法状态码
    LogEntry tooLargeStatusOutput{};

    assert(parseLogLine(
        "2026-07-31T10:15:00 INFO /login 600",
        tooLargeStatusOutput
    ) == false);

    // 测试8：日志等级不合法
    LogEntry invalidLevelOutput{};

    assert(parseLogLine(
        "2026-07-31T10:15:00 NOTICE /login 200",
        invalidLevelOutput
    ) == false);

    // 测试9：路径没有以 / 开头
    LogEntry invalidPathOutput{};

    assert(parseLogLine(
        "2026-07-31T10:15:00 INFO login 200",
        invalidPathOutput
    ) == false);

    // 测试10：合法的 WARN 日志
    LogEntry warnOutput{};

    assert(parseLogLine(
        "2026-07-31T10:15:00 WARN /login 404",
        warnOutput
    ) == true);
    assert(warnOutput.level == "WARN");
    assert(warnOutput.statusCode == 404);
    
    // 测试11：加载包含合法和非法行的日志文件
    const auto fileResult = loadLogFile(
        "../../tests/week01/sample_logs.txt"
    );

    assert(fileResult.opened == true);
    assert(fileResult.entries.size() == 3);
    assert(fileResult.invalidLineCount == 2);

    assert(fileResult.entries[0].path == "/login");
    assert(fileResult.entries[1].statusCode == 503);
    assert(fileResult.entries[2].level == "ERROR");

    // 测试12：尝试打开不存在的文件
    const auto missingFileResult = loadLogFile(
        "../../tests/week01/file_does_not_exist.txt"
    );

    assert(missingFileResult.opened == false);
    assert(missingFileResult.entries.empty());
    assert(missingFileResult.invalidLineCount == 0);

    std::cout << "All tests passed\n";
    return 0;
}
    
