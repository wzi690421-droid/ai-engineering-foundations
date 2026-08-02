#include <iostream>
#include <vector>

double calculateAverage(const std::vector<int>& values) {
    int sum = 0;

    for (const int value : values) {
        sum += value;
    }

    return sum / static_cast<double>(values.size());
}

int main() {
    const std::vector<int> values{2, 3};

    std::cout << "expected: 2.5\n";
    std::cout << "actual: " << calculateAverage(values) << '\n';
    return 0;
}
