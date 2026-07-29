#include <cstddef>
#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <functional>

std::vector<int> topKDistinct(
    const std::vector<int>& nums1,
    std::size_t k
){
    std::vector<int> nums = nums1;
   
    std::sort(nums.begin(),nums.end(),std::greater<int>());

    std::vector<int> result;

    for (int value : nums){
        if(result.empty()|| result.back() != value){
            result.push_back(value);
        }
    }

    if (result.size()>k){
        result.resize(k);
    }

    return result;
}

int main(){
    // 普通情况：包含重复数字
    std::vector<int> input1{3, 1, 5, 5, 2, 4};
    std::vector<int> expected1{5, 4, 3};

    assert(topKDistinct(input1, 3) == expected1);


    // 包含负数，而且k大于不重复元素数量
    std::vector<int> input2{-1, -5, -1, -2};
    std::vector<int> expected2{-1, -2, -5};

    assert(topKDistinct(input2, 10) == expected2);


    // 输入数组为空
    std::vector<int> input3{};
    std::vector<int> expected3{};

    assert(topKDistinct(input3, 3) == expected3);


    // k等于0，无论输入是什么，都应该返回空数组
    std::vector<int> input4{2, 2, 2};
    std::vector<int> expected4{};

    assert(topKDistinct(input4, 0) == expected4);


    std::cout << "All tests passed\n";
    return 0;
}

