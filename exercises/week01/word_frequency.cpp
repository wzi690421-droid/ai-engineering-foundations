#include <iostream>
#include <cstddef>
#include <vector>
#include <string>
#include <unordered_map>
#include <cassert>
#include <utility>
#include <algorithm>

std::unordered_map<std::string,std::size_t> countWords(
    const std::vector<std::string>& words
){
    std::unordered_map<std::string,std::size_t> counts;
    
    for(const auto& word : words) {
        ++counts[word];
    }

    return counts;
}

using WordCount = std::pair<std::string, std::size_t>;

bool wordCountComesBefore(
    const WordCount& left,
    const WordCount& right
){
    if(left.second != right.second){
        return left.second > right.second;
    }

    return left.first < right.first;
}

std::vector<WordCount> toWordCountVector(
    const std::unordered_map<std::string, std::size_t>& counts
){
    std::vector<WordCount> entries;
    entries.reserve(counts.size());

    for(const auto& [word,count] : counts){
       entries.push_back({word,count});           
    }

    std::sort(
        entries.begin(),
        entries.end(),
        wordCountComesBefore
    );

    return entries;
}

std::vector<WordCount> topKFrequent(
    const std::vector<std::string>& words,
    std::size_t k
){
    auto topfrequent = toWordCountVector(countWords(words));
    
    if (topfrequent.size()>k)
    {
        topfrequent.resize(k);
    }
    
    return topfrequent;
}

int main(){
    std::vector<std::string> input1{"cat","dog","cat","bird","dog","cat","dog","dog"};
    std::unordered_map<std::string,std::size_t> expected1{
        {"dog",4},{"cat",3},{"bird",1}
    };
    assert(countWords(input1) == expected1);

    std::vector<WordCount> expectedSorted1{
        {"dog", 4},{"cat", 3},{"bird", 1}
    };
    assert(toWordCountVector(countWords(input1)) == expectedSorted1);
    
    std::vector<WordCount> expectedTopK1{{"dog",4},{"cat",3}};
    assert(topKFrequent(input1,2) == expectedTopK1);
    
    std::vector<WordCount> expectedTopK2{};
    assert(topKFrequent(input1,0) == expectedTopK2);

    std::vector<WordCount> expectedTopK3{{"dog",4},{"cat",3},{"bird",1}};
    assert(topKFrequent(input1,10) == expectedTopK3);

    std::vector<std::string> input2{};
    std::unordered_map<std::string,std::size_t> expected2{};
    assert(countWords(input2) == expected2);

    std::vector<std::string> input3{"Cat","cat","Cat"};
    std::unordered_map<std::string,std::size_t> expected3{
        {"Cat",2},{"cat",1}     
    };
    assert(countWords(input3) == expected3);

    std::vector<std::string> input4{"dog", "apple", "dog", "apple", "cat"};
    std::vector<WordCount> expected4{
        {"apple", 2},
        {"dog", 2},
        {"cat", 1}
    };
    assert(toWordCountVector(countWords(input4)) == expected4);


    std::cout <<"All Test Passed\n";
    return 0;
}
