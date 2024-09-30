#include "model.hpp"

namespace smart {

// common funcs
long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void safe_printf(std::string piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece.empty()) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    fmt::print("{}", piece.c_str());
}


Transformer::Transformer(std::string checkpoint_path) {
    this->filename = checkpoint_path;

    // 1. get file size
    {
        std::ifstream file(checkpoint_path, std::ios::binary | std::ios::ate);
        assert(file.is_open());
        this->file_size = file.tellg();
        file.close();
    }
    // 2. load file meta data
    {
        gguf_init_params params = {
            .no_alloc = false,
            .ctx = &this->ggml_ctx
        };
        this->gguf_ctx = gguf_init_from_file(this->filename.c_str(), params);
        assert(this->gguf_ctx != nullptr);
        assert(this->ggml_ctx != nullptr);
    }
    // 3. prepare data
    {
        fill_config();
        prepare_state();
        prepare_weights();
        params = {
            .wsize = (size_t) config.dim * 32,
            .wdata = new char[config.dim * 32]
        };
    }
}

Transformer::~Transformer() {
    delete[] (char *)params.wdata;
    free_weights();
    free_state();
    gguf_free(this->gguf_ctx);
}

void Transformer::fill_config() {
    Config &c = this->config;
    gguf_context *ctx = this->gguf_ctx;
    c.dim            = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.embedding_length"));
    c.hidden_dim     = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.feed_forward_length"));
    c.n_heads        = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.attention.head_count"));
    c.n_kv_heads     = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.attention.head_count_kv"));
    c.n_layers       = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.block_count"));
    c.seq_len        = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.context_length"));
    c.vocab_size     = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.vocab_size"));
    c.rope_dim_count = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.rope.dimension_count"));    
}

void Transformer::prepare_state() {
    auto &p = this->config;
    auto &state = this->state;
    uint64_t kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    uint64_t dim = p.dim;
    uint64_t hidden_dim = p.hidden_dim;
    uint64_t n_layers = p.n_layers;
    uint64_t large_size = n_layers * p.seq_len * kv_dim;

    // alloc Optensors' buffer
    float *x           = new float[dim];
    float *xb          = new float[dim];
    float *xb2         = new float[dim];
    float *hb          = new float[hidden_dim];
    float *hb2         = new float[hidden_dim];
    float *q           = new float[dim];
    float *key_cache   = new float[large_size];
    float *value_cache = new float[large_size];
    // float *key_cache   = (float *)malloc( large_size* sizeof(float));
    // float *value_cache = (float *)malloc(large_size * sizeof(float));
    float *att         = new float[p.n_heads * p.seq_len];
    float *logits      = new float[p.vocab_size];

    // ensure all mallocs went fine
    if (!x || !xb || !xb2 || !hb || !hb2 || !q || !key_cache || !value_cache || !att || !logits) {
        fmt::println(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
    
    // alloc OpTensors
    {
        state.x = new OpTensor{.data = x, .type = GGML_TYPE_F32,
            .ne = {p.dim, 1, 1, 1},
            .nb = {sizeof(float), sizeof(float)*p.dim, sizeof(float)*p.dim, sizeof(float)*p.dim}
        };
        state.xb = new OpTensor{.data = xb, .type = GGML_TYPE_F32,
            .ne = {p.dim, 1, 1, 1},
            .nb = {sizeof(float), sizeof(float)*dim, sizeof(float)*dim, sizeof(float)*dim}
        };
        state.xb2 = new OpTensor{.data = xb2, .type = GGML_TYPE_F32,
            .ne = {p.dim, 1, 1, 1},
            .nb = {sizeof(float), sizeof(float)*dim, sizeof(float)*dim, sizeof(float)*dim}
        };
        state.hb = new OpTensor{.data = hb, .type = GGML_TYPE_F32,
            .ne = {p.hidden_dim, 1, 1, 1},
            .nb = {sizeof(float), sizeof(float)*hidden_dim, sizeof(float)*hidden_dim, sizeof(float)*hidden_dim}
        };
        state.hb2 = new OpTensor{.data = hb2, .type = GGML_TYPE_F32,
            .ne = {p.hidden_dim, 1, 1, 1},
            .nb = {sizeof(float), sizeof(float)*hidden_dim, sizeof(float)*hidden_dim, sizeof(float)*hidden_dim}
        };
        state.q = new OpTensor{.data = q, .type = GGML_TYPE_F32,
            .ne = {p.dim, 1, 1, 1},
            .nb = {sizeof(float), sizeof(float)*dim, sizeof(float)*dim, sizeof(float)*dim}
        };
        state.key_cache = new OpTensor{.data = key_cache, .type = GGML_TYPE_F32,
            .ne = {(int64_t)kv_dim, p.seq_len, p.n_layers, 1},
            .nb = {sizeof(float), sizeof(float)*kv_dim, sizeof(float)*kv_dim*p.seq_len, sizeof(float)*kv_dim*p.seq_len*n_layers}
        };
        state.value_cache = new OpTensor{.data = value_cache, .type = GGML_TYPE_F32,
            .ne = {(int64_t)kv_dim, p.seq_len, p.n_layers, 1},
            .nb = {sizeof(float), sizeof(float)*kv_dim, sizeof(float)*kv_dim*p.seq_len, sizeof(float)*kv_dim*p.seq_len*n_layers}
        };
        state.att = new OpTensor{.data = att, .type = GGML_TYPE_F32,
            .ne = {p.seq_len, p.n_heads, 1,1},
            .nb = {sizeof(float), sizeof(float)*p.seq_len, sizeof(float)*p.n_heads*p.seq_len, sizeof(float)*p.n_heads*p.seq_len}
        };
        state.logits = new OpTensor{.data = logits, .type = GGML_TYPE_F32,
            .ne = {p.vocab_size, 1, 1, 1},
            .nb = {sizeof(float), sizeof(float)*p.vocab_size, sizeof(float)*p.vocab_size, sizeof(float)*p.vocab_size}
        };
        // variable pointer, no data buffer
        state.k = new OpTensor{.data = nullptr, .type = GGML_TYPE_F32,
            .ne = {(int64_t)kv_dim, 1, 1, 1},
            .nb = {sizeof(float), sizeof(float)*kv_dim, sizeof(float)*kv_dim, sizeof(float)*kv_dim}
        };
        state.v = new OpTensor{.data = nullptr, .type = GGML_TYPE_F32,
            .ne = {(int64_t)kv_dim, 1, 1, 1},
            .nb = {sizeof(float), sizeof(float)*kv_dim, sizeof(float)*kv_dim, sizeof(float)*kv_dim}
        };

        // ensure all mallocs went fine
        if (!state.x || !state.xb || !state.xb2 || !state.hb || !state.hb2 || 
            !state.q || !state.k || !state.v || !state.key_cache || !state.value_cache || 
            !state.att || !state.logits) {
            fmt::println(stderr, "malloc optensors failed!\n");
            exit(EXIT_FAILURE);
        }

    }

}

void Transformer::free_state() {
    auto s = this->state;
    free_optensor_deep(s.x);
    free_optensor_deep(s.xb);
    free_optensor_deep(s.xb2);
    free_optensor_deep(s.hb);
    free_optensor_deep(s.hb2);
    free_optensor_deep(s.q);
    free_optensor_deep(s.att);
    free_optensor_deep(s.logits);
    free_optensor_deep(s.key_cache);
    free_optensor_deep(s.value_cache);

    free_optensor(s.k);
    free_optensor(s.v);
}

void Transformer::prepare_weights() {
    auto &w = this->weights;
    auto ctx = this->ggml_ctx;
    auto embedding = ggml_get_tensor(ctx, "token_embd.weight");
    w.token_embedding_table = get_optensor_from_ggml(embedding);
    w.output_weight         = get_optensor_from_ggml(ggml_get_tensor(ctx, "output.weight"));
    w.rms_final_weight      = get_optensor_from_ggml(ggml_get_tensor(ctx, "output_norm.weight"));
    // w.fp32_embd_table = new float[ggml_nelements(embedding)];
    
    if (embedding->type != GGML_TYPE_F32) {
        w.fp32_embd_table = new float[ggml_nelements(embedding)];
    } else {
        w.fp32_embd_table = (float *)embedding->data;
    }

    switch (embedding->type) {
        case GGML_TYPE_Q4_0:
            dequantize_row_q4_0((block_q4_0 *)w.token_embedding_table->data, w.fp32_embd_table, ggml_nelements(embedding));
            break;
        case GGML_TYPE_Q8_0:
            dequantize_row_q8_0((block_q8_0 *)w.token_embedding_table->data, w.fp32_embd_table, ggml_nelements(embedding));
            break;
        default: 
            break;
    }

    w.lw.resize(this->config.n_layers);
    for (int layer = 0; layer < this->config.n_layers; layer++) {
        w.lw[layer].attn_norm   = get_optensor_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.attn_norm.weight", layer).c_str()));
        w.lw[layer].ffn_norm    = get_optensor_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.ffn_norm.weight", layer).c_str()));
        w.lw[layer].attn_q      = get_optensor_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.attn_q.weight", layer).c_str()));
        w.lw[layer].attn_k      = get_optensor_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.attn_k.weight", layer).c_str()));
        w.lw[layer].attn_v      = get_optensor_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.attn_v.weight", layer).c_str()));
        w.lw[layer].attn_output = get_optensor_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.attn_output.weight", layer).c_str()));
        w.lw[layer].ffn_gate    = get_optensor_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.ffn_gate.weight", layer).c_str()));
        w.lw[layer].ffn_up      = get_optensor_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.ffn_up.weight", layer).c_str()));
        w.lw[layer].ffn_down    = get_optensor_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.ffn_down.weight", layer).c_str()));
    }
}

void Transformer::free_weights() {
    auto &w = this->weights;
    
    auto embedding = ggml_get_tensor(this->ggml_ctx, "token_embd.weight");
    if (embedding->type != GGML_TYPE_F32) {
        delete[] w.fp32_embd_table;
    }

    free_optensor(w.token_embedding_table);
    free_optensor(w.output_weight);
    free_optensor(w.rms_final_weight);
    

    for (int layer = 0; layer < this->config.n_layers; layer++) {
        free_optensor(w.lw[layer].attn_norm);
        free_optensor(w.lw[layer].ffn_norm);
        free_optensor(w.lw[layer].attn_q);
        free_optensor(w.lw[layer].attn_k);
        free_optensor(w.lw[layer].attn_v);
        free_optensor(w.lw[layer].attn_output);
        free_optensor(w.lw[layer].ffn_gate);
        free_optensor(w.lw[layer].ffn_up);
        free_optensor(w.lw[layer].ffn_down);
    }
    w.lw.clear();
}

void Transformer::multihead_attention(
    OpTensor* q_tensor, 
    OpTensor* k_cache_tensor, 
    OpTensor* attn_tensor, 
    OpTensor* v_cache_tensor, 
    OpTensor* xb_tensor, 
    uint32_t pos, 
    uint64_t loff) 
{
    auto p = &this->config;
    auto dim = p->dim;
    auto kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    auto kv_mul = p->n_heads / p->n_kv_heads;
    auto head_size = dim / p->n_heads;

    uint32_t h = 0;
    for (h = 0; h < p->n_heads; h++) {
        auto q   = (float *)q_tensor->data + h *head_size;
        auto att = (float *)attn_tensor->data + h * p->seq_len;
        
        for (auto t = 0; t <= pos; t++) {
            auto k = (float *)k_cache_tensor->data + loff + t * kv_dim + (h / kv_mul) * head_size;
            auto score = 0.0f;
            
            for (auto i = 0; i < head_size; i++) {
                score += q[i] * k[i];
            }
        
            score /= sqrtf(head_size);
            att[t] = score;
        }

        softmax_internal(att, pos + 1);

        auto xb = (float *)xb_tensor->data + h * head_size;
        memset(xb, 0, head_size * sizeof(float));

        for (auto t = 0; t <= pos; t++) {
            auto v = (float *)v_cache_tensor->data + loff + t * kv_dim + (h / kv_mul) * head_size;
            auto a = att[t];

            for (auto i = 0; i < head_size; i++) {
                xb[i] += a * v[i];
            }
        
        }

    }
}

void rope(
    uint32_t dim, 
    uint32_t head_size, 
    int pos, 
    uint32_t kv_dim, 
    OpTensor *q_tensor,
    OpTensor *k_tensor) 
{

    for (int i = 0; i < dim; i+=2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
        for (int v = 0; v < rotn; v++) {
            float* vec = v == 0 ? (float *)q_tensor->data : (float *)k_tensor->data; // the vector to rotate (query or key)
            float v0 = vec[i];
            float v1 = vec[i+1];
            vec[i]   = v0 * fcr - v1 * fci;
            vec[i+1] = v0 * fci + v1 * fcr;
        }
    }
}

float* Transformer::forward(int token, int pos) {
    auto p = &this->config;
    auto w = &this->weights;
    auto s = &this->state;

    auto dim = p->dim;
    auto kv_dim =  (p->dim * p->n_kv_heads) / p->n_heads;
    auto kv_mul = p->n_heads / p->n_kv_heads;
    auto hidden_dim =  p->hidden_dim;
    auto head_size = dim / p->n_heads;

    float* content_row = w->fp32_embd_table + token * dim;
    memcpy(s->x->data, content_row, dim*sizeof(float));

    for(auto L = 0; L < p->n_layers; L++) {
        rmsnorm(s->xb, s->x, w->lw[L].attn_norm);

        uint64_t loff = L * p->seq_len * kv_dim;
        s->k->data = (float*)s->key_cache->data + loff + pos * kv_dim;
        s->v->data = (float*)s->value_cache->data + loff + pos * kv_dim;

        ggml_compute_forward_op_mul_mat(&params, s->q, w->lw[L].attn_q, s->xb);
        ggml_compute_forward_op_mul_mat(&params, s->k, w->lw[L].attn_k, s->xb);
        ggml_compute_forward_op_mul_mat(&params, s->v, w->lw[L].attn_v, s->xb);

        rope(dim, head_size, pos, kv_dim, s->q, s->k);

        multihead_attention(s->q, s->key_cache, s->att, s->value_cache, s->xb, pos, loff);

        ggml_compute_forward_op_mul_mat(&params, s->xb2, w->lw[L].attn_output, s->xb);

        // residual connection
        for (auto i = 0; i < dim; i++) {
            ((float*)s->x->data)[i] += ((float *)s->xb2->data)[i];
        }

        rmsnorm(s->xb, s->x, w->lw[L].ffn_norm);

        ggml_compute_forward_op_mul_mat(&params, s->hb, w->lw[L].ffn_gate, s->xb);
        ggml_compute_forward_op_mul_mat(&params, s->hb2, w->lw[L].ffn_up, s->xb);

        for (auto i = 0; i < hidden_dim; i++) {
            float val = ((float *)s->hb->data)[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= ((float *)s->hb2->data)[i];
            ((float *)s->hb->data)[i] = val;
        }

        ggml_compute_forward_op_mul_mat(&params, s->xb, w->lw[L].ffn_down, s->hb);
        
        // residual connection
        for (int i = 0; i < dim; i++) {
            ((float *)s->x->data)[i] += ((float *)s->xb->data)[i];
        }

    }

    rmsnorm(s->x, s->x, w->rms_final_weight);

    ggml_compute_forward_op_mul_mat(&params, s->logits, w->output_weight, s->x);

    return (float *)s->logits->data;
}

void Transformer::generate(LlamaTokenizer *tk, Sampler *sampler, std::string prompt, int steps) {

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    auto prompt_tokens = tk->tokenize(prompt, true);
    num_prompt_tokens = prompt_tokens.size();
    
    if (num_prompt_tokens < 1) {
        fmt::println(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }
    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    auto token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forward(token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sampler->sample(logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS token delimits sequences
        if (next == tk->bos_token()) { 
            break; 
        }

        // print the token as string, decode it with the Tokenizer object
        auto piece = tk->to_string(next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
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
        fmt::println(stderr, "achieved tok/s: {}\n", (pos-1) / (double)(end-start)*1000);
    }
}

void Transformer::debug_config_info() {
    auto &c = this->config;
    fmt::println("dim       :{:6}", c.dim);
    fmt::println("hidden_dim:{:6}", c.hidden_dim);
    fmt::println("n_heads   :{:6}", c.n_heads);
    fmt::println("n_kv_heads:{:6}", c.n_kv_heads);
    fmt::println("n_layers  :{:6}", c.n_layers);
    fmt::println("seq_len   :{:6}", c.seq_len);
    fmt::println("vocab_size:{:6}", c.vocab_size);
}

void Transformer::debug_weights_info() {
    auto &w = this->weights;
    debug_weight_info("token embd", w.token_embedding_table);
    debug_weight_info("rms output", w.rms_final_weight);
    debug_weight_info("output", w.output_weight);
    int layer = 0;
    for (auto &l: w.lw) {
        debug_weight_info(fmt::format("[{}]attn_norm", layer), l.attn_norm);
        debug_weight_info(fmt::format("[{}]ffn_norm", layer), l.ffn_norm);
        debug_weight_info(fmt::format("[{}]attn_q", layer), l.attn_q);
        debug_weight_info(fmt::format("[{}]attn_k", layer), l.attn_k);
        debug_weight_info(fmt::format("[{}]attn_v", layer), l.attn_v);
        debug_weight_info(fmt::format("[{}]attn_output", layer), l.attn_output);
        debug_weight_info(fmt::format("[{}]ffn_gate", layer), l.ffn_gate);
        debug_weight_info(fmt::format("[{}]ffn_up", layer), l.ffn_up);
        debug_weight_info(fmt::format("[{}]ffn_down", layer), l.ffn_down);
        layer += 1;
    }
}

} // namespace smart