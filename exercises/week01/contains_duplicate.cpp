#include <cstddef>
#include <iostream>
#include <vector>
#include <cassert>

bool containsDuplicate(const std::vector<int>& nums){
    for(std::size_t i=0;i<nums.size();i++){
       for (std::size_t j=i+1;j<nums.size();j++){
        if(nums[i] == nums[j]){
        return true;
        }
       }
    }
    return false;
}

int main(){
    std::vector<int> input1 {1,2,3,1};
    assert(containsDuplicate(input1) == true);

    std::vector<int> input2 {1,2,3};
    assert(containsDuplicate(input2) == false);

    std::vector<int> input3 {};
    assert(containsDuplicate(input3) == false);

    std::vector<int> input4 {-1,-1};
    assert(containsDuplicate(input4) == true);

    std::cout<<"all test passed"<<"\n";
    return 0;
}