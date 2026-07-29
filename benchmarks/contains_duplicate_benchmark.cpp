#include <cassert>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include <vector>

// 将结果写入volatile变量，避免编译器因为“结果没有被使用”
// 而直接删除被测函数的执行过程。
volatile bool benchmarkSink = false;

bool containsDuplicateQuadratic(const std::vector<int>& nums) {
    for (std::size_t i = 0; i < nums.size(); ++i) {
        for (std::size_t j = i + 1; j < nums.size(); ++j) {
            if (nums[i] == nums[j]) {
                return true;
            }
        }
    }

    return false;
}

bool containsDuplicateHash(const std::vector<int>& nums) {
    std::unordered_set<int> seen;

    for (int value : nums) {
        if (seen.find(value) != seen.end()) {
            return true;
        }

        seen.insert(value);
    }

    return false;
}

bool containsDuplicateHashReserved(const std::vector<int>& nums) {
    std::unordered_set<int> seen;
    seen.reserve(nums.size());

    for (int value : nums) {
        if (seen.find(value) != seen.end()) {
            return true;
        }

        seen.insert(value);
    }

    return false;
}

template <typename Function>
double measureMilliseconds(
    Function function,
    const std::vector<int>& data,
    int repeats
) {
    const auto start = std::chrono::steady_clock::now();

    bool accumulatedResult = false;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        accumulatedResult ^= function(data);
    }

    const auto finish = std::chrono::steady_clock::now();
    benchmarkSink = accumulatedResult;

    const auto elapsed =
        std::chrono::duration<double, std::milli>(finish - start).count();

    return elapsed / repeats;
}

int main() {
    constexpr std::size_t size = 8000;

    // 场景A：完全没有重复元素。
    std::vector<int> uniqueData(size);
    std::iota(uniqueData.begin(), uniqueData.end(), 0);

    // 场景B：前两个元素立即重复。
    std::vector<int> earlyDuplicateData = uniqueData;
    earlyDuplicateData[1] = earlyDuplicateData[0];

    // 场景C：最后两个元素才重复。
    std::vector<int> lateDuplicateData = uniqueData;
    lateDuplicateData[size - 1] = lateDuplicateData[size - 2];

    std::cout
        << "scenario,n,quadratic_ms,hash_ms,hash_reserved_ms,"
        << "hash_div_reserved\n"
        << std::fixed
        << std::setprecision(6);

    const auto runScenario = [size](
        const char* scenario,
        const std::vector<int>& data,
        bool expected,
        int repeats
    ) {
        const bool quadraticResult = containsDuplicateQuadratic(data);
        const bool hashResult = containsDuplicateHash(data);
        const bool hashReservedResult = containsDuplicateHashReserved(data);
        assert(quadraticResult == hashResult);
        assert(hashResult == hashReservedResult);
        assert(hashResult == expected);

        benchmarkSink = containsDuplicateQuadratic(data);
        benchmarkSink = containsDuplicateHash(data);
        benchmarkSink = containsDuplicateHashReserved(data);

        const double quadraticMs =
            measureMilliseconds(containsDuplicateQuadratic, data, repeats);
        const double hashMs =
            measureMilliseconds(containsDuplicateHash, data, repeats);
        const double hashReservedMs =
            measureMilliseconds(containsDuplicateHashReserved, data, repeats);

        const double reserveSpeedup = hashMs / hashReservedMs;

        std::cout
            << scenario << ','
            << size << ','
            << quadraticMs << ','
            << hashMs << ','
            << hashReservedMs << ','
            << reserveSpeedup << '\n';
    };

    // 运行时间较长的场景重复5次；极快的开头重复场景多重复，
    // 避免单次时间太短而完全被计时噪声淹没。
    runScenario("A_unique", uniqueData, false, 5);
    runScenario("B_early_duplicate", earlyDuplicateData, true, 10001);
    runScenario("C_late_duplicate", lateDuplicateData, true, 5);

    return 0;
}
