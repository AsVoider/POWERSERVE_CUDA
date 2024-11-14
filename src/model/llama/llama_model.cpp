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
#include "tokenizer/tokenizer.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace smart {

LlamaModel::LlamaModel(
    const std::string &filename, const std::shared_ptr<Config> &config, const std::shared_ptr<Platform> &platform
) :
    Model(filename) {
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
    m_platform = platform;

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

auto LlamaModel::forward(
    const std::vector<int> &tokens, const std::vector<int> &pos, const CausalAttentionMask &mask, bool lm_head
) -> std::vector<std::vector<float>> {
    Graph g;
    // input embedding
    size_t batch_size = tokens.size();
    auto embd_tb      = g.add_tensor(m_weights->token_embedding_table);
    auto x            = g.get_embedding(embd_tb, tokens);
    Tensor *logits    = nullptr;

#if defined(SMART_WITH_QNN)
    if (m_platform->qnn_backend) {
        logits = g.qnn_forward(x, pos, mask, m_config->tf_cfg.vocab_size, lm_head);
    } else
#endif

    {
        SMART_UNUSED(lm_head);
        // attention and ffn
        for (size_t L = 0; L < m_config->tf_cfg.n_layers; L++) {
            auto att_o = m_attn->build(g, x, L, pos, mask);
            auto ffn_o = m_ffn->build(g, att_o, L);
            x          = ffn_o;
        }

        // final output
        auto rms_final_w    = g.add_tensor(m_weights->rms_final_weight);
        auto final_rms_norm = g.rms_norm(x, rms_final_w, m_config->tf_cfg.norm_eps);

        auto output_w = g.add_tensor(m_weights->output_weight);
        logits        = g.mat_mul(final_rms_norm, output_w);
    }

    Executor executor(*m_platform, g);
    executor.allocate_buffers();

    executor.run();
    float *logits_data = static_cast<float *>(logits->get<ggml::Buffer>().m_data);
    auto res           = std::vector<std::vector<float>>();
    if (lm_head) {
        for (size_t i = 0; i < batch_size; i++) {
            res.emplace_back(std::vector<float>(
                logits_data + i * m_config->tf_cfg.vocab_size, logits_data + (i + 1) * m_config->tf_cfg.vocab_size
            ));
        }
    }

    return res;
}

void LlamaModel::generate(Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps) {
    // encode the (string) prompt into tokens sequence

    int num_prompt_tokens = 0;
    auto prompt_tokens    = tokenizer.tokenize(prompt, tokenizer.m_vocab.tokenizer_add_bos);
    // fmt::println("tokens: {}", prompt_tokens);
    num_prompt_tokens = prompt_tokens.size();

    SMART_ASSERT(num_prompt_tokens >= 1);
    // start the main loop
    long start = 0; // used to time our code, only initialized after first iteration
    int next;       // will store the next token in the sequence
    int pos = 0;    // position in the sequence
#if defined(SMART_WITH_QNN)
    if (m_platform->qnn_backend) {
        pos = m_platform->qnn_backend->m_causalLM->kv_cache->position;
    }
#endif
    //prefill
    auto prefill_start = time_in_ms();
    auto prefill_pos   = std::vector<int>(num_prompt_tokens - 1);
    std::iota(prefill_pos.begin(), prefill_pos.end(), pos);
    auto prefill_attention_mask = CausalAttentionMask(num_prompt_tokens - 1);
    forward(
        std::vector<int>(prompt_tokens.begin(), std::prev(prompt_tokens.end())),
        prefill_pos,
        prefill_attention_mask,
        false
    );
    fmt::print("{}", prompt);
    auto prefill_end           = time_in_ms();
    auto token                 = prompt_tokens[num_prompt_tokens - 1]; // kick off with the first token in the prompt
    auto decode_attention_mask = CausalAttentionMask(1);
#if defined(SMART_WITH_QNN)
    if (m_platform->qnn_backend) {
        pos = m_platform->qnn_backend->m_causalLM->kv_cache->position;
    } else
#endif
    {
        pos = num_prompt_tokens - 1;
    }
    while (pos < steps) {
        // forward the transformer to get logits for the next token
        std::vector<float> logits =
            forward(std::vector<int>(1, token), std::vector<int>(1, pos), decode_attention_mask)[0];

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
        fmt::println(
            "prefill speed: {} tokens/s", (num_prompt_tokens - 1) / (double)(prefill_end - prefill_start) * 1000
        );
        fmt::println(stderr, "decode speed: {} tokens/s", (pos - num_prompt_tokens) / (double)(end - start) * 1000);
        fmt::println(stderr, "total speed: {} tokens/s", (pos - 1) / (double)(end - prefill_start) * 1000);
    }
}

} // namespace smart
