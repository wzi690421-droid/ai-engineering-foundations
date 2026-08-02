#include <iostream>

int calculateScore(int base, int bonus) {
    return base * 2 + bonus;
}

int main() {
    std::cout << calculateScore(10, 5) << '\n';
    return 0;
}
