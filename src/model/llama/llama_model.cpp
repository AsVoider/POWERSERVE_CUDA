#include "llama_model.hpp"

#include "backend/ggml/buffer.hpp"
#include "backend/platform.hpp"
#include "common.hpp"
#include "executor/executor.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "model/llama/llama_config.hpp"
#include "model/llama/llama_weight.hpp"
#include "sampler/sampler.hpp"
#include "sampler/sampler_chain.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace smart {

LlamaModel::LlamaModel(const std::string &filename, int n_threads) : Model(filename) {
    // load file meta data (+ 4G)
    {
        gguf_init_params params = {.no_alloc = false, .ctx = &ggml_ctx};
        gguf_ctx                = gguf_init_from_file(filename.c_str(), params);
        SMART_ASSERT(gguf_ctx != nullptr);
        SMART_ASSERT(ggml_ctx != nullptr);
    }
    // prepare data
    m_config = std::make_shared<LlamaConfig>(gguf_ctx);
    // prepare weights (+ 2G)
    m_weights = std::make_shared<LlamaWeight>(ggml_ctx, m_config->tf_cfg.n_layers, m_config->tf_cfg.dim);
    // modules
    m_attn = nullptr;
    m_ffn  = std::make_shared<FFN>(m_config, m_weights);
    // platform
    plat = std::make_shared<Platform>(m_config, n_threads);

    // debug model info
    {
        ggml::debug_system_info();
        // ggml::debug_meta_info(gguf_ctx, ggml_ctx);
        // m_config->debug_config_info();
        // ggml::debug_tensors_info(gguf_ctx, ggml_ctx);
    }
}

LlamaModel::~LlamaModel() {
    gguf_free(gguf_ctx);
}

Graph *LlamaModel::prefill() {
    return nullptr;
}

Graph *LlamaModel::decode() {
    return nullptr;
}

std::vector<float> LlamaModel::forward(int token, int pos) {
    Graph g;
    auto dim = m_config->tf_cfg.dim;

    // input embedding
    // prepare input : embeding token tensor [dim,]
    SMART_ASSERT(token * dim + dim <= m_weights->fp32_embd_table.size());
    auto x                  = g.new_tensor(DataType::FP32, {dim});
    TensorNode *tensor_embd = x;
    auto pos_tensor         = g.new_tensor(DataType::INT32, {1});
    // attention and ffn
    for (size_t L = 0; L < m_config->tf_cfg.n_layers; L++) {
        auto att_o = m_attn->build(g, x, L, pos_tensor, pos);
        auto ffn_o = m_ffn->build(g, att_o, L);
        x          = ffn_o;
    }

    // final output
    auto rms_final_w    = g.add_tensor(m_weights->rms_final_weight);
    auto final_rms_norm = g.rms_norm(x, rms_final_w);

    auto output_w = g.add_tensor(m_weights->output_weight);
    auto logits   = g.mat_mul(final_rms_norm, output_w);

    Executor executor(*plat.get(), g);
    executor.allocate_buffers();
    memcpy(
        tensor_embd->get<ggml::Buffer>().m_data,
        static_cast<void *>(m_weights->fp32_embd_table.data() + token * dim),
        dim * sizeof(float)
    );
    static_cast<int32_t *>(pos_tensor->get<ggml::Buffer>().m_data)[0] = pos;

    executor.run();
    float *logits_data = static_cast<float *>(logits->get<ggml::Buffer>().m_data);

    return std::vector<float>(logits_data, logits_data + m_config->tf_cfg.vocab_size);
}

void LlamaModel::generate(Tokenizer *tk, Sampler *sampler, std::string prompt, int steps) {
    SMART_ASSERT(m_attn != nullptr);
    // encode the (string) prompt into tokens sequence

    int num_prompt_tokens = 0;
    auto prompt_tokens    = tk->tokenize(prompt, tk->m_vocab.tokenizer_add_bos);
    num_prompt_tokens     = prompt_tokens.size();

    SMART_ASSERT(num_prompt_tokens >= 1);
    // start the main loop
    long start = 0;                // used to time our code, only initialized after first iteration
    int next;                      // will store the next token in the sequence
    auto token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos    = 0;                // position in the sequence
    while (pos < steps) {
        // forward the transformer to get logits for the next token
        // float* logits = forward(token, pos);
        std::vector<float> logits = forward(token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // TODO: prefill
            // if we are still processing the input prompt, force the next prompt token
            next = tk->m_vocab.tokenizer_add_bos ? prompt_tokens[pos + 1] : prompt_tokens[pos];
        } else {
            // TODO: Decode
            // otherwise sample the next token from the logits
            auto probs = ProbArray(logits);
            sampler->apply(probs);
            std::mt19937 gen(std::random_device{}());
            next = probs.sample(gen).index;
            ((SamplerChain *)sampler)->accept(next);
        }
        pos++;

        // data-dependent terminating condition: the BOS token delimits sequences
        if (next == tk->bos_token()) {
            break;
        } else if (next == tk->m_vocab.special_eos_id || next == tk->m_vocab.special_eom_id || next == tk->m_vocab.special_eot_id) {
            fmt::print("[end of text]");
            break;
        }

        // print the token as string, decode it with the Tokenizer object
        auto piece = tk->to_string(next);
        fmt::print("{}", piece);
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) {
            start = time_in_ms();
        }
    }
    fmt::println("");

    if (pos > 1) {
        long end = time_in_ms();
        fmt::println(stderr, "achieved tok/s: {}\n", (pos - 1) / (double)(end - start) * 1000);
    }
}

} // namespace smart
