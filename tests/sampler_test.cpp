#include <vector>
#include <cmath>
#include <iostream>

#include "sampler/sampler_chain.hpp"
#include "common.hpp"

using namespace smart;
int main() {
    std::vector<float> logits{0.1, 0.9, 0.2, 0.4, 0.7, 0.8, 0.1, 0.6, 0.2, 0.02};
    std::vector<float> logits2{1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0};
    {
        size_t ans = 1;
        auto p = convert_logits(logits);
        auto g = greedy_sample(p);
        // std::cout << g.index << ":" << g.prob << std::endl;
        SMART_ASSERT(g.index == ans);
    }
    {
        size_t topk = 4;
        std::vector<size_t> ans{1, 5, 4, 7};

        auto p = convert_logits(logits);
        // for(auto l: p) {
        //     std::cout << l.index << ":" << l.prob << ", ";
        // }
        // std::cout << std::endl;

        topk_sample(p, topk);
        // for(auto l: p) {
        //     std::cout << l.index << ":" << l.prob << ", ";
        // }
        // std::cout << std::endl;
        for (int i = 0; i < topk; i++) {
            SMART_ASSERT(p[i].index == ans[i]);
        }
    }
    {
        std::vector<float> ans{0.0236405, 0.0642617, 0.174681, 0.474833, 0.0236405, 0.0642617, 0.174681};
        auto p2 = convert_logits(logits2);
        apply_temperature(p2, 1.);
        // for(auto l: p2) {
        //     std::cout << l.index << ":" << l.prob << ", ";
        // }
        // std::cout << std::endl;
        for (int i = 0; i < logits2.size(); i++) {
            SMART_ASSERT(std::abs(p2[i].prob - ans[i]) < 1e-6);
        }
    }

    fmt::println("All tests passed!");
}