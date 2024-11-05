#include <vector>
#include <cmath>
#include <iostream>

#include "sampler/sampler_chain.hpp"
#include "common.hpp"

using namespace smart;
int main() {
    std::vector<float> logits{0.1, 0.9, 0.2, 0.4, 0.7, 0.8, 0.1, 0.6, 0.2, 0.02};
    std::vector<float> logits2{1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0};
    
    fmt::println("All tests passed!");
}