#include "internvl_model.hpp"

#include "backend/ggml/buffer.hpp"
#include "common/logger.hpp"
#include "common/type_def.hpp"
#include "executor/executor.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "model/llama/llama_weight.hpp"
#include "process_image_internvl2.hpp"
#include "sampler/sampler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace smart {

InternVL::InternVL(const std::string &filename, const std::shared_ptr<ModelConfig> &config) : Model(filename) {
    {
        gguf_init_params params = {.no_alloc = false, .ctx = &ggml_ctx};
        gguf_ctx                = gguf_init_from_file(filename.c_str(), params);
        SMART_ASSERT(gguf_ctx != nullptr);
        SMART_ASSERT(ggml_ctx != nullptr);
    }
    m_config  = config;
    lazy_load = ggml_get_tensor(ggml_ctx, "output.weight") == nullptr ? true : false;
    m_weights = std::make_shared<LlamaWeight>(ggml_ctx, m_config->llm.n_layers, lazy_load);
    if (lazy_load) {
        SMART_LOG_WARN("only the embedding table was loaded");
    }
    m_ffn = std::make_shared<FFN>(m_config->llm, m_weights);
}

InternVL::~InternVL() {
    gguf_free(gguf_ctx);
}

auto InternVL::preprocess(const std::vector<Path> &img_paths, const std::string &prompt) -> std::string {
    InternVL2ImageProcessor img_processor;
    for (auto &img_path : img_paths) {
        fmt::println("\nload image: {}", img_path);
        pixel_values_list.push_back(img_processor.load_image(img_path));
    }
    // static constexpr const char *img_end = "</img>\n";
    static constexpr const char *img_pad   = "<IMG_CONTEXT>";
    static constexpr const char *img_start = "<img>";
    const auto img_pixel_size = m_config->vision.image_size * m_config->vision.image_size * m_config->vision.in_chans;
    std::string processed_prompt = "<|im_start|>user\n" + prompt;
    size_t img_pos               = 0;
    size_t img_idx               = 0;
    int img_tokens_length        = 0;
    while ((img_pos = processed_prompt.find("<img>", img_pos)) != std::string::npos) {
        const auto num_patch = pixel_values_list[img_idx].size() / img_pixel_size;
        img_infos.push_back({num_patch, 0});
        img_tokens_length = num_patch * m_config->vision.num_tokens_per_patch;
        img_pos += strlen(img_start);
        std::string img_context;
        for (int context_idx = 0; context_idx < img_tokens_length; context_idx++) {
            img_context += img_pad;
        }
        processed_prompt.insert(img_pos, img_context);
        img_pos += img_context.size();
        img_idx++;
    }
    assert(img_idx == pixel_values_list.size());
    processed_prompt += "<|im_end|><|im_start|>assistant\n";
    return processed_prompt;
}

auto InternVL::forward(
    const std::vector<int> &tokens, const std::vector<int> &pos, const CausalAttentionMask &mask, bool lm_head
) -> std::vector<std::vector<float>> {
    Graph g(m_config->model_id);
    // prompt embedding
    size_t batch_size = tokens.size();
    auto embd_tb      = g.add_tensor(m_weights->token_embedding_table);
    auto x            = g.get_embedding(embd_tb, tokens);
    Tensor *logits    = nullptr;
    int img_idx       = 0;
    for (size_t token_idx = 0; token_idx < tokens.size(); token_idx++) {
        if (tokens[token_idx] == IMG_START) {
            img_infos[img_idx].second = (token_idx + 1) * m_config->vision.embed_dim;
        }
    }

#if defined(SMART_WITH_QNN)
    if (m_platform->qnn_backend) {
        if (pixel_values_list.empty()) {
            logits = g.qnn_forward(x, pos, mask, m_config->llm.vocab_size, lm_head);
        } else {
            logits = g.qnn_forward_vl(x, pos, mask, m_config->llm.vocab_size, lm_head, pixel_values_list, img_infos);
        }

    } else
#endif

    {
        SMART_UNUSED(lm_head);
        SMART_UNUSED(pos);
        SMART_UNUSED(x);
        SMART_UNUSED(mask);
        fmt::println("Internvl Model not support in cpu");
        SMART_ASSERT(false);
    }

    Executor executor(*m_platform, g);
    executor.allocate_buffers();

    executor.run();
    float *logits_data = static_cast<float *>(logits->get<ggml::Buffer>().m_data);
    auto res           = std::vector<std::vector<float>>();
    if (lm_head) {
        for (size_t i = 0; i < batch_size; i++) {
            res.emplace_back(std::vector<float>(
                logits_data + i * m_config->llm.vocab_size, logits_data + (i + 1) * m_config->llm.vocab_size
            ));
        }
    }

    return res;
}

auto InternVL::decode(Sampler &sampler, const std::vector<Token> tokens, const std::vector<int> pos, bool lm_head)
    -> std::vector<Token> {
    auto mask = CausalAttentionMask(tokens.size());
    auto ret  = forward(tokens, pos, mask, lm_head);
    std::vector<Token> toks;
    for (auto logits : ret) {
        auto probs = ProbArray(logits);
        sampler.apply(probs);
        std::mt19937 gen(std::random_device{}());
        auto next = probs.sample(gen).index;
        sampler.accept(next);
        toks.push_back(next);
    }
    return toks;
}

auto InternVL::generate(Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps, size_t batch_size)
    -> Model::TokenRange {
    std::vector<Path> imgs;
    size_t start_pos = 0, end_pos = 0;
    std::string start_tag   = "<img>";
    std::string end_tag     = "</img>";
    std::string replacement = "<img></img>";
    std::string instruction(prompt);

    while ((start_pos = instruction.find(start_tag, start_pos)) != std::string::npos) {
        start_pos += start_tag.size();
        end_pos = instruction.find(end_tag, start_pos);
        if (end_pos == std::string::npos)
            break;
        imgs.push_back(Path(instruction.substr(start_pos, end_pos - start_pos)));

        instruction.replace(
            start_pos - start_tag.size(), (end_pos + end_tag.size()) - (start_pos - start_tag.size()), replacement
        );

        start_pos = start_pos - start_tag.size() + replacement.size(); // Move past the <image> tag
    }

    std::string processed_prompt = prompt;
    if (!imgs.empty()) {
        processed_prompt = preprocess(imgs, instruction);
    }
    return Model::TokenRange(*this, tokenizer, sampler, processed_prompt, steps, batch_size);
}

} // namespace smart
