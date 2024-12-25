#include "speculator.hpp"

#include "common/perf.hpp"

#include <algorithm>

namespace smart {

std::vector<int> Speculative::TokenTree::spec_sample(std::vector<float> &logits, int topk) {
    std::vector<int> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Use nth_element to partition the topk elements
    std::nth_element(indices.begin(), indices.begin() + topk, indices.end(), [&logits](int a, int b) {
        return logits[a] > logits[b];
    });

    // Sort the topk elements
    std::sort(indices.begin(), indices.begin() + topk, [&logits](int a, int b) { return logits[a] > logits[b]; });

    // Filter the results based on the condition
    std::vector<int> filtered_indices;
    for (int i = 0; i < topk; ++i) {
        filtered_indices.push_back(indices[i]);
    }

    return filtered_indices;
}

void Speculative::TokenTree::build_tree(
    Token father, int father_id, int now_depth, long &elapsed_time, double now_prob
) {
    if (now_prob < 0.3)
        return;
    this_turn_depth = std::max(this_turn_depth, now_depth);
    int id          = __idx++;
    SMART_ASSERT(id <= MAX_SPEC_NODES, "too many nodes");

    if (father_id != -1) {
        son[father_id].push_back(id);
    }
    SMART_ASSERT(id == int(tk_list.size()), "wrong index");

    tk_list.push_back(father);
    depth_list.push_back(now_depth);
    fa_list.push_back(father_id);

    const TimeCounter model_forward_timer;
    auto logits = m_model->forward({father}, {m_previous_position + now_depth}, CausalAttentionMask(1));
    elapsed_time += model_forward_timer.get_time_in_ms();

    auto x = logits[0];
    std::nth_element(x.begin(), x.begin() + MAX_EXPANSION, x.end(), std::greater<float>());
    std::vector<float> topk_logits(x.begin(), x.begin() + MAX_EXPANSION);

    ProbArray probs(topk_logits);
    probs.softmax();

    std::nth_element(
        probs.m_probs.begin(), probs.m_probs.begin() + MAX_EXPANSION, probs.m_probs.end(), std::greater<ProbIndex>()
    );
    auto probs_indices = probs.m_probs;
    std::sort(probs_indices.begin(), probs_indices.end(), std::greater<ProbIndex>());

    int expansion = 1;

    if (now_depth >= MAX_DEPTH) {
        if (now_depth > MAX_DEPTH * 3)
            return;
        if (now_prob < 0.5)
            return;
    }

    auto tokens = spec_sample(logits[0], expansion);

    int cnt = 0;
    for (auto token : tokens) {
        auto prob = probs.m_probs[cnt++].prob;
        build_tree(token, id, now_depth + 1, elapsed_time, prob * now_prob);
    }
}

void Speculative::generate(Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps) {
    m_tokenizer = &tokenizer;

    int num_prompt_tokens = 0;
    auto prompt_tokens    = tokenizer.tokenize(prompt, tokenizer.m_vocab.tokenizer_add_bos);
    num_prompt_tokens     = prompt_tokens.size();

    SMART_ASSERT(num_prompt_tokens >= 1);
    // start the main loop
    long start = 0; // used to time our code, only initialized after first iteration
    int next;       // will store the next token in the sequence
    int pos = 0;    // position in the sequence
#if defined(SMART_WITH_QNN)
    if (m_target_model->m_platform->qnn_backend) {
        pos = m_target_model->m_platform->qnn_backend->m_causal_vlm->kv_cache->position;
    }
#endif
    //prefill
    auto prefill_start = time_in_ms();
    auto prefill_pos   = std::vector<int>(num_prompt_tokens - 1);
    std::iota(prefill_pos.begin(), prefill_pos.end(), pos);
    auto prefill_attention_mask = CausalAttentionMask(num_prompt_tokens - 1);
    m_target_model->forward(
        std::vector<int>(prompt_tokens.begin(), std::prev(prompt_tokens.end())),
        prefill_pos,
        prefill_attention_mask,
        false
    );
    // long temp_start = time_in_ms();
    m_draft_model->forward(
        std::vector<int>(prompt_tokens.begin(), std::prev(prompt_tokens.end())),
        prefill_pos,
        prefill_attention_mask,
        false
    );

    fmt::print("{}", prompt);
    auto prefill_end = time_in_ms();

    auto token                 = prompt_tokens[num_prompt_tokens - 1]; // kick off with the first token in the prompt
    auto decode_attention_mask = CausalAttentionMask(1);
#if defined(SMART_WITH_QNN)
    if (m_target_model->m_platform->qnn_backend) {
        pos = m_target_model->m_platform->qnn_backend->m_causal_vlm->kv_cache->position;
    } else
#endif
    {
        pos = num_prompt_tokens - 1;
    }

    next = token;
    std::vector<std::string> output;
    output.push_back(prompt);
    int draft_depth = m_draft_depth;

    while (pos < steps) {

        TokenTree tk_tree(m_draft_model, m_tokenizer, draft_depth, pos, m_expansion);

        long elapsed_time = 0;
        tk_tree.build_tree(next, elapsed_time);
        stats.draft_time += elapsed_time;

        int bs = tk_tree.tk_list.size();
        std::vector<int> token_vec(bs);
        std::vector<int> pos_vec(bs, 0);
        for (int i = 0; i < bs; i++) {
            pos_vec[i]   = tk_tree.depth_list[i] + pos;
            token_vec[i] = tk_tree.tk_list[i];
        }

        std::vector<std::vector<bool>> mask(bs, std::vector<bool>(bs, 0));

        for (int i = 0; i < bs; i++) {
            if (tk_tree.fa_list[i] >= 0)
                mask[i] = mask[tk_tree.fa_list[i]];
            mask[i][i] = true;
        }

        CausalAttentionMask batch_mask(bs, mask);

        long temp_start                        = time_in_ms();
        std::vector<std::vector<float>> logits = m_target_model->forward(token_vec, pos_vec, batch_mask);
        long temp_end                          = time_in_ms();
        stats.target_time += temp_end - temp_start;

        std::vector<int> pos_to_foward;
        std::vector<int> tk_to_foward;

        int n_accepted = 0;
        int cache_pos  = pos;
        for (int local_id = 0; local_id < (int)bs;) {
            auto logit = logits[local_id];
            auto probs = ProbArray(logit);
            sampler.apply(probs);
            std::mt19937 gen(std::random_device{}());
            next = probs.sample(gen).index;

            bool verified = false;
            for (auto i : tk_tree.son[local_id]) {
                if (tk_tree.tk_list[i] == next) {
                    verified = true;
                    local_id = i;
                    cache_pos++;
                    m_target_model->m_platform->qnn_backend->m_causal_vlm->kv_cache->move(cache_pos, pos + i);
                    m_draft_model->m_platform->qnn_backend->m_causal_vlm->kv_cache->move(cache_pos, pos + i);
                    break;
                }
            }

            auto piece = tokenizer.to_string(next);
            fmt::print("{}", piece);
            output.push_back(piece);
            fflush(stdout);

            // verified = false;

            if (verified) {
                n_accepted++;
                // draft_model->forward({next}, {pos + n_accepted}, CausalAttentionMask(1))[0];
                pos_to_foward.push_back(pos + n_accepted);
                tk_to_foward.push_back(next);

                sampler.accept(next);
                if (next == tokenizer.bos_token()) {
                    break;
                } else if (next == tokenizer.m_vocab.special_eos_id || next == tokenizer.m_vocab.special_eom_id ||
                           next == tokenizer.m_vocab.special_eot_id) {
                    break;
                }
            } else
                break;
        }
        m_target_model->m_platform->qnn_backend->m_causal_vlm->kv_cache->rollback(bs - n_accepted - 1);
        m_draft_model->m_platform->qnn_backend->m_causal_vlm->kv_cache->rollback(bs - n_accepted - 1);
        pos += n_accepted + 1;
        stats.all_tk_num += draft_depth;
        stats.accept_tk_num += n_accepted;
        stats.accept_tk_num_every_turn.push_back(n_accepted);
        stats.accept_rate_every_turn.push_back(n_accepted / (double)tk_tree.this_turn_depth);
        stats.draft_depth_every_turn.push_back(tk_tree.this_turn_depth);

        // data-dependent terminating condition: the BOS token delimits sequences
        if (next == tokenizer.bos_token()) {
            break;
        } else if (next == tokenizer.m_vocab.special_eos_id || next == tokenizer.m_vocab.special_eom_id ||
                   next == tokenizer.m_vocab.special_eot_id) {
            fmt::print("[end of text]");
            break;
        }

        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) {
            start = time_in_ms();
        }
    }

    if (pos > 1) {
        int system_prompt_num = 11;
        long end              = time_in_ms();
        SMART_LOG_INFO(
            "\n\nprefill speed: {} tokens/s", (num_prompt_tokens - 1) / (double)(prefill_end - prefill_start) * 1000
        );
        SMART_LOG_INFO(
            "decode speed: {} tokens/s", (pos - num_prompt_tokens - system_prompt_num) / (double)(end - start) * 1000
        );
        SMART_LOG_INFO(
            "accept rate: {}%",
            100.0 * std::accumulate(stats.accept_tk_num_every_turn.begin(), stats.accept_tk_num_every_turn.end(), 0) /
                (double)stats.accept_tk_num_every_turn.size() /
                (std::accumulate(stats.draft_depth_every_turn.begin(), stats.draft_depth_every_turn.end(), 0) /
                 (double)stats.draft_depth_every_turn.size())
        );
        SMART_LOG_INFO(
            "average generate toks every turn: {} / {}",
            std::accumulate(stats.accept_tk_num_every_turn.begin(), stats.accept_tk_num_every_turn.end(), 0) /
                    (double)stats.accept_tk_num_every_turn.size() +
                1,
            std::accumulate(stats.draft_depth_every_turn.begin(), stats.draft_depth_every_turn.end(), 0) /
                    (double)stats.draft_depth_every_turn.size() +
                1
        );
        SMART_LOG_INFO(
            "draft time: {} s, {}% of all model time",
            stats.draft_time / 1000.0,
            stats.draft_time * 100.0 / (stats.draft_time + stats.target_time)
        );
        SMART_LOG_INFO(
            "target time: {} s, {}% of all model time",
            stats.target_time / 1000.0,
            100.0 - stats.draft_time * 100.0 / (stats.draft_time + stats.target_time)
        );
        SMART_LOG_INFO(
            "npu free time: {} s, {}% of all decoding time",
            (end - prefill_end - stats.draft_time - stats.target_time) / 1000.0,
            (end - prefill_end - stats.draft_time - stats.target_time) * 100.0 / (end - prefill_end)
        );
        // 统计Accept数量各自是多少
        std::map<int, int> accept_tk_num_every_turn;
        for (auto i : stats.accept_tk_num_every_turn) {
            accept_tk_num_every_turn[i]++;
        }
        SMART_LOG_INFO("Accept_tk_num_every_turn:");
        for (auto [k, v] : accept_tk_num_every_turn) {
            SMART_LOG_INFO("-- Accept_tk_num_every_turn[{}]: {}", k, v);
        }
        SMART_LOG_INFO("All accept_tk_num_every_turn: {}", stats.accept_tk_num_every_turn);
        SMART_LOG_INFO("Accept rate every turn: [");
        for (auto i : stats.accept_rate_every_turn) {
            SMART_LOG_INFO("{:.2f}, ", i);
        }
        SMART_LOG_INFO("]");
        SMART_LOG_INFO("Draft depth every turn: {}", stats.draft_depth_every_turn);
        SMART_LOG_INFO(
            "Prefilled tokens: {}, Decoded tokens: {}",
            num_prompt_tokens - 1,
            pos - num_prompt_tokens - system_prompt_num
        );
    }
}
}; // namespace smart
