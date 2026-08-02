#include <cstddef>
#include <iostream>
#include <vector>

int sumValues(const std::vector<int>& values) {
    int sum = 0;

    for (std::size_t index = 0; index < values.size(); ++index) {
        sum += values[index];
    }

    return sum;
}

int main() {
    const std::vector<int> values{10, 20, 30};
    std::cout << sumValues(values) << '\n';
    return 0;
}
