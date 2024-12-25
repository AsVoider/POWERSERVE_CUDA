// training-free, general speculative
// training-free, general speculative
// training-free, general speculative
#pragma once

#include "model/model.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace smart {

struct Speculative {

    static const int MAX_EXPANSION_LAYER = 80, MAX_SPEC_NODES = 80;

    using Token = llama_vocab::id;

    std::unique_ptr<smart::Model> m_target_model, m_draft_model;
    std::vector<int> m_expansion;
    std::string m_model_arch;
    int m_draft_depth;

    Tokenizer *m_tokenizer;

    struct TokenTree {
        const int MAX_EXPANSION = 500, MAX_DEPTH = 15;
        const std::unique_ptr<smart::Model> &m_model;
        smart::Tokenizer *m_tokenizer;
        int m_draft_depth, m_previous_position, this_turn_depth = 0;
        std::vector<int> &m_expansion;

        std::vector<Token> tk_list, verified_list, fa_list; // store the tokens in the tree as a list(like an array)
        std::vector<int> son[MAX_SPEC_NODES], depth_list;   // store the sons of every node

        TokenTree(
            const std::unique_ptr<smart::Model> &model,
            smart::Tokenizer *tk,
            int draft_depth,
            int prepos,
            std::vector<int> &expansion
        ) :
            m_model(model),
            m_tokenizer(tk),
            m_draft_depth(draft_depth),
            m_previous_position(prepos),
            m_expansion(expansion) {}

        std::vector<int> spec_sample(std::vector<float> &logits, int topk = 3);

        // the pos is the position of the latest token that was not forwarded(i.e. get from the last logits)
        void build_tree(Token root, long &elaspsed_time) {
            // __build_son(-1, 0, 0);
            __idx = 0;
            build_tree(root, -1, 0, elaspsed_time, 1);
        }

        void build_tree(Token father, int father_id, int now_depth, long &temp_time, double now_prob);

        int __idx = 0;
    };

    struct Stats {
        int accept_tk_num = 0;
        int all_tk_num    = 0;
        int batch_size    = 0;
        std::vector<int> accept_tk_num_every_turn;
        std::vector<double> accept_rate_every_turn;
        std::vector<int> draft_depth_every_turn;
        long draft_time = 0, target_time = 0;
    } stats;

    Speculative(
        std::unique_ptr<smart::Model> main_model,
        std::unique_ptr<smart::Model> draft_model,
        std::vector<int> expansion = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
    ) :
        m_target_model(std::move(main_model)),
        m_draft_model(std::move(draft_model)) {
        this->m_expansion = expansion;
        m_draft_depth     = expansion.size();
        // generate
    }

    ~Speculative() = default;

    // void generate(smart::Tokenizer *tk, smart::Sampler *sampler, std::string prompt, int steps);
    void generate(Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps);
};

} // namespace smart
