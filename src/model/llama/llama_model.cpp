#include "llama_model.hpp"

#include "backend/ggml/buffer.hpp"
#include "backend/platform.hpp"
#include "common.hpp"
#include "executor/executor.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
// #include "model/llama/llama_config.hpp"
#include "model/llama/llama_weight.hpp"
#include "sampler/sampler.hpp"
#include "sampler/sampler_chain.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace smart {

LlamaModel::LlamaModel(const std::string &filename, std::shared_ptr<Config> config) : Model(filename) {
    // load file meta data (+ 4G)
    {
        gguf_init_params params = {.no_alloc = false, .ctx = &ggml_ctx};
        gguf_ctx                = gguf_init_from_file(filename.c_str(), params);
        SMART_ASSERT(gguf_ctx != nullptr);
        SMART_ASSERT(ggml_ctx != nullptr);
    }
    // prepare data
    m_config = config;
    // prepare weights (+ 2G)
    m_weights = std::make_shared<LlamaWeight>(ggml_ctx, m_config->tf_cfg.n_layers);
    // modules
    m_attn = nullptr;
    m_ffn  = std::make_shared<FFN>(m_config, m_weights);
    // platform
    m_plat = nullptr;

    // debug model info
    {
        // ggml::debug_system_info();
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

auto LlamaModel::forward(int token, int pos) -> std::vector<float> {
    Graph g;
    // input embedding
    size_t batch_size = 1;
    auto tokens       = g.new_tensor(DataType::INT32, {batch_size});
    auto embd_tb      = g.add_tensor(m_weights->token_embedding_table);
    auto x            = g.get_embedding(embd_tb, tokens);
    auto pos_tensor   = g.new_tensor(DataType::INT32, {1, batch_size});
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

    Executor executor(*m_plat.get(), g);
    executor.allocate_buffers();

    for (size_t i = 0; i < batch_size; i++) {
        // TODO: support batch
        static_cast<int32_t *>(tokens->get<ggml::Buffer>().m_data)[i]     = token;
        static_cast<int32_t *>(pos_tensor->get<ggml::Buffer>().m_data)[i] = pos;
    }

    executor.run();
    float *logits_data = static_cast<float *>(logits->get<ggml::Buffer>().m_data);
    // TODO: support batch
    return std::vector<float>(logits_data, logits_data + m_config->tf_cfg.vocab_size);
}

void LlamaModel::generate(Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps) {
    SMART_ASSERT(m_attn != nullptr);
    // encode the (string) prompt into tokens sequence

    int num_prompt_tokens = 0;
    auto prompt_tokens    = tokenizer.tokenize(prompt, tokenizer.m_vocab.tokenizer_add_bos);
    // fmt::println("tokens: {}", prompt_tokens);
    num_prompt_tokens = prompt_tokens.size();

    SMART_ASSERT(num_prompt_tokens >= 1);
    // start the main loop
    long start = 0;                // used to time our code, only initialized after first iteration
    int next;                      // will store the next token in the sequence
    auto token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos    = 0;                // position in the sequence
    while (pos < steps) {
        // forward the transformer to get logits for the next token
        std::vector<float> logits = forward(token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // TODO: prefill
            // if we are still processing the input prompt, force the next prompt token
            next = tokenizer.m_vocab.tokenizer_add_bos ? prompt_tokens[pos + 1] : prompt_tokens[pos];
        } else {
            // TODO: Decode
            // otherwise sample the next token from the logits
            auto probs = ProbArray(logits);
            sampler.apply(probs);
            std::mt19937 gen(std::random_device{}());
            next = probs.sample(gen).index;
            sampler.accept(next);
        }
        pos++;

        // data-dependent terminating condition: the BOS token delimits sequences
        if (next == tokenizer.bos_token()) {
            break;
        } else if (next == tokenizer.m_vocab.special_eos_id || next == tokenizer.m_vocab.special_eom_id ||
                   next == tokenizer.m_vocab.special_eot_id) {
            fmt::print("[end of text]");
            break;
        }

        // print the token as string, decode it with the Tokenizer object
        auto piece = tokenizer.to_string(next);
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
