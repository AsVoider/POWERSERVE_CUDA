// training-free, general speculative
// training-free, general speculative
// training-free, general speculative
#pragma once

#include "common.hpp"
#include "model/llama/llama_model.hpp"
#include "model/module/norm_attention.hpp"
#include "model/module/quest_attention.hpp"
#include "sampler/sampler_chain.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstdlib>
#include <memory>
#include <string>

namespace smart {

struct Speculative {

    static const int MAX_EXPANSION_LAYER = 80, MAX_SPEC_NODES = 80;

    using Token = llama_vocab::id;

    std::shared_ptr<smart::Model> m_draft_model, m_target_model;
    std::vector<int> m_expansion;
    std::string m_model_arch;
    int m_draft_depth;

    Tokenizer *m_tokenizer;

    struct TokenTree {
        const int MAX_EXPANSION = 500, MAX_DEPTH = 15;
        std::shared_ptr<smart::Model> m_model;
        smart::Tokenizer *m_tokenizer;
        int m_draft_depth, m_previous_position, this_turn_depth = 0;
        std::vector<int> &m_expansion;

        std::vector<Token> tk_list, verified_list, fa_list; // store the tokens in the tree as a list(like an array)
        std::vector<int> son[MAX_SPEC_NODES], depth_list;   // store the sons of every node

        TokenTree(
            std::shared_ptr<smart::Model> model,
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

    std::shared_ptr<smart::Model> qnn_model_init(
        std::string file_path, std::string config_path, std::string qnn_path, std::string attn_type = "normal"
    ) {
        int n_threads = 4;
        // get number of CPUs
        {
            auto n_cpus = uv_available_parallelism(); // Default fallback value
            // NOTE: the main thread is also a worker thread, so we need to subtract 1
            SMART_ASSERT(n_cpus >= 2);
            n_threads = std::min((unsigned int)n_threads, n_cpus - 1);
        }

        // get config
        auto config = std::make_shared<smart::Config>(config_path);

        // get platform
        auto platform = std::make_shared<smart::Platform>();
        platform->init_ggml_backend(config, n_threads);
#if defined(SMART_WITH_QNN)
        if (qnn_path != "") {
            platform->init_qnn_backend(qnn_path, config);
        }
#endif

        // get model type
        std::string model_arch = config->arch;
        smart::get_memory_usage("begin");

        std::shared_ptr<smart::Model> model;
        // TODO: move into Model.cpp like build_model
        if (model_arch == "llama") {
            // convert model to llama
            model = std::make_shared<smart::LlamaModel>(file_path, config, platform);
            if (m_model_arch == "")
                m_model_arch = "llama";
            else if (m_model_arch != "llama")
                SMART_ASSERT(false);
        } else if (model_arch == "phi3") {
            SMART_ASSERT(false);
        } else {
            fmt::print("Unknown model type\n");
        }
        smart::get_memory_usage("after model init");

        if (attn_type == "normal") {
            model->m_attn = std::make_shared<smart::NormAttention>(model->m_config, model->m_weights);
        } else if (attn_type == "quest") {
            model->m_attn = std::make_shared<smart::QuestAttention>(model->m_config, model->m_weights);
            // SMART_ASSERT(false);
        }
        smart::get_memory_usage("after attn init");

        return model;
    }

    Speculative(
        std::string target_path,
        std::string draft_path,
        std::string target_config_path,
        std::string draft_config_path,
        std::string target_qnn_path,
        std::string draft_qnn_path,
        std::vector<int> expansion = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
    ) {
        m_target_model = qnn_model_init(target_path, target_config_path, target_qnn_path);
        m_draft_model  = qnn_model_init(draft_path, draft_config_path, draft_qnn_path);

        this->m_expansion = expansion;
        m_draft_depth     = expansion.size();
        // generate
    }

    ~Speculative() = default;

    // void generate(smart::Tokenizer *tk, smart::Sampler *sampler, std::string prompt, int steps);
    void generate(Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps);
};

} // namespace smart
