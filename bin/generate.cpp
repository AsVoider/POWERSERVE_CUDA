#include <cassert>

#include "generate.hpp"
#include "ggml.h"

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

void rmsnorm_internal(float* o, float* x, float* weight, int64_t size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void rmsnorm(OpTensor *o, OpTensor *x, OpTensor *weight) {
    assert(o != nullptr && o->type == GGML_TYPE_F32 && o->data != nullptr);
    assert(x != nullptr && x->type == GGML_TYPE_F32 && x->data != nullptr);
    assert(weight != nullptr && weight->type == GGML_TYPE_F32 && weight->data != nullptr);

    auto size = x->ne[0];

    rmsnorm_internal((float *)o->data, (float *)x->data, (float *)weight->data, size);
}

void softmax_internal(float* x, int64_t size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void softmax(OpTensor *x, int64_t size) {
    assert(x != nullptr && x->type == GGML_TYPE_F32 && x->data != nullptr);
    softmax_internal((float *)x->data, size);
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void multihead_attention(
    Config* p, 
    OpTensor* q_tensor, 
    OpTensor* k_cache_tensor, 
    OpTensor* attn_tensor, 
    OpTensor* v_cache_tensor, 
    OpTensor* xb_tensor, 
    uint32_t pos, 
    uint64_t loff) 
{
    auto dim = p->dim;
    auto kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    auto kv_mul = p->n_heads / p->n_kv_heads;
    auto head_size = dim / p->n_heads;

    uint32_t h = 0;
    #pragma omp parallel for private(h)
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

float* forward(Transformer* tf, int token, int pos) {
    auto p = &tf->config;
    auto w = &tf->weights;
    auto s = &tf->state;

    auto x_tensor = s->x;
    auto a = s->xb;
    auto a2 = s->xb2;
    auto k_tensor = s->k;
    auto q_tensor = s->q;
    auto v_tensor = s->v;
    auto hb_tensor = s->hb;
    auto hb2_tensor = s->hb2;
    auto kcache_tensor = s->key_cache;
    auto vcache_tensor = s->value_cache;
    auto attn_tensor = s->att;
    auto logits_tensor = s->logits;

    auto dim = p->dim;
    auto kv_dim =  (p->dim * p->n_kv_heads) / p->n_heads;
    auto kv_mul = p->n_heads / p->n_kv_heads;
    auto hidden_dim =  p->hidden_dim;
    auto head_size = dim / p->n_heads;

    float* content_row = w->fp32_embd_table + token * dim;
    memcpy(s->x->data, content_row, dim*sizeof(float));

    // TODO: temp buffer for compute, not use hard-coded
    void *w_data = malloc(dim * 16 * p->vocab_size);
    struct op_compute_params params = {
        .wsize = (size_t) dim * 16 * p->vocab_size,
        .wdata = w_data
    };

    for(auto L = 0; L < p->n_layers; L++) {
        rmsnorm(a, x_tensor, tf->weights.lw[L].attn_norm);

        uint64_t loff = L * p->seq_len * kv_dim;
        k_tensor->data = (float*)kcache_tensor->data + loff + pos * kv_dim;
        v_tensor->data = (float*)vcache_tensor->data + loff + pos * kv_dim;

        ggml_compute_forward_op_mul_mat(&params, q_tensor, tf->weights.lw[L].attn_q, a);
        ggml_compute_forward_op_mul_mat(&params, k_tensor, tf->weights.lw[L].attn_k, a);
        ggml_compute_forward_op_mul_mat(&params, v_tensor, tf->weights.lw[L].attn_v, a);

        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? (float *)s->q->data : (float *)s->k->data; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        multihead_attention(p, q_tensor, kcache_tensor, attn_tensor, vcache_tensor, a, pos, loff);

        ggml_compute_forward_op_mul_mat(&params, a2, tf->weights.lw[L].attn_output, a);

        // residual connection
        for (auto i = 0; i < dim; i++) {
            ((float*)x_tensor->data)[i] += ((float *)a2->data)[i];
        }

        rmsnorm(a, x_tensor, w->lw[L].ffn_norm);

        ggml_compute_forward_op_mul_mat(&params, hb_tensor, w->lw[L].ffn_gate, a);
        ggml_compute_forward_op_mul_mat(&params, hb2_tensor, w->lw[L].ffn_up, a);

        for (auto i = 0; i < hidden_dim; i++) {
            float val = ((float *)hb_tensor->data)[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= ((float *)hb2_tensor->data)[i];
            ((float *)hb_tensor->data)[i] = val;
        }

        ggml_compute_forward_op_mul_mat(&params, a, w->lw[L].ffn_down, hb_tensor);
        
        // residual connection
        for (int i = 0; i < dim; i++) {
            ((float *)x_tensor->data)[i] += ((float *)a->data)[i];
        }

    }

    rmsnorm(x_tensor, x_tensor, w->rms_final_weight);

    ggml_compute_forward_op_mul_mat(&params, logits_tensor, w->output_weight, x_tensor);
    free(w_data);

    return (float *)logits_tensor->data;
}

void generate(Transformer *tf, smart::LlamaTokenizer *tk, Sampler *sampler, std::string prompt, int steps) {

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
        float* logits = forward(tf, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
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
        fmt::println(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }
}

} // namespace smart