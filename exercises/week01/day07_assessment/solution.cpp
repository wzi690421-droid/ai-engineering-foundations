#include <cstddef>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <cassert>

struct Request {
    std::string path;
    int statusCode;
};

using PathCount = std::pair<std::string, std::size_t>;

struct RequestSummary{
    std::size_t totalCount{};
    std::size_t serverErrorCount{};
    std::vector<PathCount> topPaths;
};

RequestSummary summarizeRequests(
    const std::vector<Request>& requests,
    std::size_t k
) {
    RequestSummary summary{};
    std::unordered_map<std::string, std::size_t> pathCounts;

    summary.totalCount = requests.size();

    for (const Request& request : requests) {
        if (request.statusCode >= 500 && request.statusCode < 600) {
            ++summary.serverErrorCount;
        }
        if (request.statusCode >= 200 && request.statusCode < 400) {
            ++pathCounts[request.path];
        }
    }

    summary.topPaths.reserve(pathCounts.size());

    for (const auto& [path, count] : pathCounts) {
            summary.topPaths.push_back({path, count});
    }

    std::sort(summary.topPaths.begin(), summary.topPaths.end(),
              [](const PathCount& left, const PathCount& right) {
                  if (left.second != right.second) {
                      return left.second > right.second;
                  }
                  return left.first < right.first;
              });

    if (summary.topPaths.size() > k) {
        summary.topPaths.resize(k);
    }

    return summary;
}

int main() {

    std::vector<Request> requests1 = {
        {"/login", 200},
        {"/login", 500},
        {"/health", 503},
        {"/login", 200},
        {"/health", 200},
        {"/login", 503}
    };
    RequestSummary summary1 = summarizeRequests(requests1, 2);
    assert(summary1.totalCount == 6);
    assert(summary1.serverErrorCount == 3);
    assert((summary1.topPaths == std::vector<PathCount>{{"/login", 2},{"/health", 1}}));

    std::vector<Request> requests2 = {};
    RequestSummary summary2 = summarizeRequests(requests2, 2);
    assert(summary2.totalCount == 0);
    assert(summary2.serverErrorCount == 0);
    assert((summary2.topPaths == std::vector<PathCount>{}));


    RequestSummary summary3 = summarizeRequests(requests1, 0);
    assert(summary3.totalCount == 6);
    assert(summary3.serverErrorCount == 3);
    assert((summary3.topPaths == std::vector<PathCount>{}));

    RequestSummary summary4 = summarizeRequests(requests1, 8);
    assert(summary4.totalCount == 6);
    assert(summary4.serverErrorCount == 3);
    assert((summary4.topPaths == std::vector<PathCount>{{"/login", 2}, {"/health", 1}}));

    RequestSummary summary5 = summarizeRequests(requests1, 3);
    assert(summary5.totalCount == 6);
    assert(summary5.serverErrorCount == 3);
    assert((summary5.topPaths == std::vector<PathCount>{{"/login", 2}, {"/health", 1}}));

    std::vector<Request> requests6 = {
        {"/login", 499},
        {"/login", 500},
        {"/health", 599},
        {"/health", 600},
    };
    RequestSummary summary6 = summarizeRequests(requests6, 2);
    assert(summary6.totalCount == 4);
    assert(summary6.serverErrorCount == 2);
    assert((summary6.topPaths == std::vector<PathCount>{}));

    std::vector<Request> requests7 = {
        {"/login", 199},
        {"/login", 200},
        {"/health", 399},
        {"/health", 400},
    };
    RequestSummary summary7 = summarizeRequests(requests7, 2);
    assert(summary7.totalCount == 4);
    assert(summary7.serverErrorCount == 0);
    assert((summary7.topPaths == std::vector<PathCount>{ {"/health", 1},{"/login", 1}}));

    std::cout << "Day 7 part 2 passed\n";

    return 0;
}
