#include <cstddef>
#include <iostream>
#include <vector>
#include <cassert>
#include <unordered_set>

bool containsDuplicate(const std::vector<int>& nums){
     std::unordered_set<int> seen;

     for(int value : nums){
        if(seen.find(value) != seen.end()){
            return true;
        }
        seen.insert(value);
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