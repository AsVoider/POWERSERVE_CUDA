#include <cassert>
#include <string>
#include <vector>

#include "fmt/base.h"
#include "fmt/format.h"
#include "ggml.h"
#include "llama_tokenizer.hpp"

// --------------------
// key structs
struct Config{
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
};

// typedef struct {
//     // token embedding table
//     float* token_embedding_table;    // (vocab_size, dim)
//     // weights for rmsnorms
//     float* rms_att_weight; // (layer, dim) rmsnorm weights
//     float* rms_ffn_weight; // (layer, dim)
//     // weights for matmuls. note dim == n_heads * head_size
//     float* wq; // (layer, dim, n_heads * head_size)
//     float* wk; // (layer, dim, n_kv_heads * head_size)
//     float* wv; // (layer, dim, n_kv_heads * head_size)
//     float* wo; // (layer, n_heads * head_size, dim)
//     // weights for ffn
//     float* w1; // (layer, hidden_dim, dim)
//     float* w2; // (layer, dim, hidden_dim)
//     float* w3; // (layer, hidden_dim, dim)
//     // final rmsnorm
//     float* rms_final_weight; // (dim,)
//     // (optional) classifier weights for the logits, on the last layer
//     float* wcls;
// } TransformerWeights;

struct RunState{ // -> OpTensor
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
    // quantized xq and hq
};

struct LayerWeights {
    OpTensor *attn_norm;   // "blk.$.attn_norm.weight"
    OpTensor *ffn_norm;    // "blk.$.ffn_norm.weight"
    
    OpTensor *attn_q;      // "blk.$.attn_q.weight"
    OpTensor *attn_k;      // "blk.$.attn_k.weight"
    OpTensor *attn_v;      // "blk.$.attn_v.weight"
    OpTensor *attn_output; // "blk.$.attn_output.weight"

    OpTensor *ffn_gate;    // "blk.$.ffn_gate.weight"
    OpTensor *ffn_up;      // "blk.$.ffn_up.weight"
    OpTensor *ffn_down;    // "blk.$.ffn_down.weight"
};

struct TransformerWeights {
    OpTensor *token_embedding_table; // "token_embd.weight"
    OpTensor *wcls;                  // "output.weight"
    OpTensor *rms_att_weight;        // "output_norm.weight"
    float * fp32_embd_table;

    std::vector<LayerWeights> lw;
};

struct Transformer{
    Config config;              // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state;             // buffers for the "wave" of activations in the forward pass

    std::string filename;  // file descriptor for memory mapping
    ssize_t file_size;     // size of the checkpoint file in bytes
    gguf_context *gguf_ctx = nullptr;
    ggml_context *ggml_ctx = nullptr;
};

#define QK8_0 32

struct block_q8_0{
    uint16_t d;       // delta
    int8_t  qs[QK8_0]; // quants
};

void dequantize_row_q8_0(const block_q8_0 *x, float * y, int64_t k) {
    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

OpTensor *Tensor2OpTensor(ggml_tensor *t) {
    OpTensor *opt = new OpTensor({
        .data = t->data,
        .type = t->type
    });
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        opt->ne[i] = t->ne[i];
        opt->nb[i] = t->nb[i];
    }
    return opt;
}

void fill_config(Transformer *t) {
    Config &c = t->config;
    int idx = 0;

    idx = gguf_find_key(t->gguf_ctx, "llama.embedding_length");
    c.dim = gguf_get_val_u32(t->gguf_ctx, idx);

    idx = gguf_find_key(t->gguf_ctx, "llama.feed_forward_length");
    c.hidden_dim = gguf_get_val_u32(t->gguf_ctx, idx);

    idx = gguf_find_key(t->gguf_ctx, "llama.attention.head_count");
    c.n_heads = gguf_get_val_u32(t->gguf_ctx, idx);

    idx = gguf_find_key(t->gguf_ctx, "llama.attention.head_count_kv");
    c.n_kv_heads = gguf_get_val_u32(t->gguf_ctx, idx);

    idx = gguf_find_key(t->gguf_ctx, "llama.block_count");
    c.n_layers = gguf_get_val_u32(t->gguf_ctx, idx);

    idx = gguf_find_key(t->gguf_ctx, "llama.context_length");
    c.seq_len = gguf_get_val_u32(t->gguf_ctx, idx);

    idx = gguf_find_key(t->gguf_ctx, "llama.vocab_size");
    c.vocab_size = gguf_get_val_u32(t->gguf_ctx, idx);
}

void prepare_state(Transformer *t) {
    auto &p = t->config;
    auto &state = t->state;
    int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    state.x = (float *)calloc(p.dim, sizeof(float));
    state.xb = (float *)calloc(p.dim, sizeof(float));
    state.xb2 = (float *)calloc(p.dim, sizeof(float));
    state.hb = (float *)calloc(p.hidden_dim, sizeof(float));
    state.hb2 = (float *)calloc(p.hidden_dim, sizeof(float));
    state.q = (float *)calloc(p.dim, sizeof(float));
    state.key_cache = (float *)calloc(p.n_layers * p.seq_len * kv_dim, sizeof(float));
    state.value_cache = (float *)calloc(p.n_layers * p.seq_len * kv_dim, sizeof(float));
    state.att = (float *)calloc(p.n_heads * p.seq_len, sizeof(float));
    state.logits = (float *)calloc(p.vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!state.x || !state.xb || !state.xb2 || !state.hb || !state.hb2 || !state.q
     || !state.key_cache || !state.value_cache || !state.att || !state.logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_state(Transformer *t) {
    auto &state = t->state;
    free(state.x);
    free(state.xb);
    free(state.xb2);
    free(state.hb);
    free(state.hb2);
    free(state.q);
    free(state.att);
    free(state.logits);
    free(state.key_cache);
    free(state.value_cache);
}

void prepare_weights(Transformer *t) {
    auto &w = t->weights;
    auto &ggml_ctx = t->ggml_ctx;
    w.token_embedding_table = Tensor2OpTensor(ggml_get_tensor(ggml_ctx, "token_embd.weight"));
    w.wcls = Tensor2OpTensor(ggml_get_tensor(ggml_ctx, "output.weight"));
    w.rms_att_weight = Tensor2OpTensor(ggml_get_tensor(ggml_ctx, "output_norm.weight"));
    w.fp32_embd_table = (float *)calloc(sizeof(float), ggml_nelements(ggml_get_tensor(ggml_ctx, "token_embd.weight")));
    dequantize_row_q8_0((block_q8_0 *)w.token_embedding_table->data, w.fp32_embd_table, ggml_nelements(ggml_get_tensor(ggml_ctx, "token_embd.weight")));

    w.lw.resize(t->config.n_layers);
    for (int layer = 0; layer < t->config.n_layers; layer++) {
        w.lw[layer].attn_norm   = Tensor2OpTensor(ggml_get_tensor(ggml_ctx, fmt::format("blk.{}.attn_norm.weight", layer).c_str()));
        w.lw[layer].ffn_norm    = Tensor2OpTensor(ggml_get_tensor(ggml_ctx, fmt::format("blk.{}.ffn_norm.weight", layer).c_str()));
        w.lw[layer].attn_q      = Tensor2OpTensor(ggml_get_tensor(ggml_ctx, fmt::format("blk.{}.attn_q.weight", layer).c_str()));
        w.lw[layer].attn_k      = Tensor2OpTensor(ggml_get_tensor(ggml_ctx, fmt::format("blk.{}.attn_k.weight", layer).c_str()));
        w.lw[layer].attn_v      = Tensor2OpTensor(ggml_get_tensor(ggml_ctx, fmt::format("blk.{}.attn_v.weight", layer).c_str()));
        w.lw[layer].attn_output = Tensor2OpTensor(ggml_get_tensor(ggml_ctx, fmt::format("blk.{}.attn_output.weight", layer).c_str()));
        w.lw[layer].ffn_gate    = Tensor2OpTensor(ggml_get_tensor(ggml_ctx, fmt::format("blk.{}.ffn_gate.weight", layer).c_str()));
        w.lw[layer].ffn_up      = Tensor2OpTensor(ggml_get_tensor(ggml_ctx, fmt::format("blk.{}.ffn_up.weight", layer).c_str()));
        w.lw[layer].ffn_down    = Tensor2OpTensor(ggml_get_tensor(ggml_ctx, fmt::format("blk.{}.ffn_down.weight", layer).c_str()));
    }
}

void free_layer_weight(LayerWeights &lw) {
    delete lw.attn_norm;
    delete lw.ffn_norm;
    delete lw.attn_q;
    delete lw.attn_k;
    delete lw.attn_v;
    delete lw.attn_output;
    delete lw.ffn_gate;
    delete lw.ffn_up;
    delete lw.ffn_down;
}

void free_weights(Transformer *t) {
    for (auto l: t->weights.lw) {
        free_layer_weight(l);
    }
    delete t->weights.token_embedding_table;
    delete t->weights.wcls;
    delete t->weights.rms_att_weight;
}


void build_transformer(Transformer *t, std::string checkpoint_path) {
    // get file size
    {
        std::ifstream file(checkpoint_path, std::ios::binary | std::ios::ate);
        assert(file.is_open());
        t->file_size = file.tellg();
        file.close();
    }

    gguf_init_params params = {
        .no_alloc = false,
        .ctx = &t->ggml_ctx
    };
    t->filename = checkpoint_path;
    t->gguf_ctx = gguf_init_from_file(t->filename.c_str(), params);
    assert(t->gguf_ctx != nullptr);

    fill_config(t);
    prepare_state(t);
    prepare_weights(t);
}

void free_transformer(Transformer* t) {
    free_weights(t);
    free_state(t);
    gguf_free(t->gguf_ctx);
}

// --------------------
// Sampler
typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = (ProbIndex *)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    next = sample_argmax(logits, sampler->vocab_size);
    return next;
}

// --------------------
// global variables
std::string file_path = "/home/zwb/SS/models/Llama-2-7b-chat-hf/llama-2-7b.Q8_0.gguf";
std::string tokenizer_path = "/home/zwb/SS/models/Llama-2-7b-chat-hf/llama2_7b_vocab.gguf";
float temperature = 1.0f;                // 0.0 = greedy deterministic. 1.0 = original. don't set higher
float topp = 0.9f;                       // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
int steps = 64;                          // number of steps to run for
std::string prompt = "One day,";         // prompt string
unsigned long long rng_seed = 2024927;   // seed rng with time by default

// --------------------------------
// key funcs

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
    printf("%s", piece.c_str());
}

void rmsnorm(float* o, float* x, float* weight, int size) {
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

void softmax(float* x, int size) {
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

float* forward(Transformer* tf, int token, int pos) {
    Config *p = &tf->config;
    TransformerWeights *w = &tf->weights;
    RunState *s = &tf->state;
    
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // TODO: Need Q8_0 type
    float* content_row = w->fp32_embd_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));


    // blk = 32 * int8 + float16 scale
    // add to run state
    int q8_0_size = 2;
    void* w_data = malloc(dim * 16 * p->vocab_size);

    struct op_compute_params params = {
        .wsize = (size_t) dim * 16 * p->vocab_size,
        .wdata = w_data,
    };

    struct OpTensor a = {
        .data = s->xb, // -> OpTensor *
        .type = GGML_TYPE_F32,
        .ne = {dim, 1,1,1},
        .nb = {sizeof(float), sizeof(float)*dim, sizeof(float)*dim, sizeof(float)*dim}
        // .nb = {ggml_type_size(GGML_TYPE_Q8_0), ggml_type_size(GGML_TYPE_Q8_0)*dim/GROUP_Q8_0, ggml_type_size(GGML_TYPE_Q8_0)*dim/GROUP_Q8_0, ggml_type_size(GGML_TYPE_Q8_0)*dim/GROUP_Q8_0}
    };

    struct OpTensor q_tensor = {
        .data = (void *) s->q,
        .type = GGML_TYPE_F32,
        .ne = {dim, 1,1,1},
        .nb = {sizeof(float), sizeof(float)*dim, sizeof(float)*dim, sizeof(float)*dim}
    };
    struct OpTensor k_tensor = {
        .data = (void *) s->k,
        .type = GGML_TYPE_F32,
        .ne = {dim, 1, 1,1},
        .nb = {sizeof(float), sizeof(float)*dim, sizeof(float)*dim, sizeof(float)*dim}
    };
    struct OpTensor v_tensor = {
        .data = (void *) s->v,
        .type = GGML_TYPE_F32,
        .ne = {dim, 1,1,1},
        .nb = {sizeof(float), sizeof(float)*dim, sizeof(float)*dim, sizeof(float)*dim}
    };
    struct OpTensor a2 = {
        .data = (void *) s->xb2,
        .type = GGML_TYPE_F32,
        .ne = {dim, 1,1,1},
        .nb = {sizeof(float), sizeof(float)*dim, sizeof(float)*dim, sizeof(float)*dim}
    };

    struct OpTensor hb_tensor = {
        .data = (void *) s->hb,
        .type = GGML_TYPE_F32,
        .ne = {hidden_dim, 1,1,1},
        .nb = {sizeof(float), sizeof(float)*hidden_dim, sizeof(float)*hidden_dim, sizeof(float)*hidden_dim}
    };

    struct OpTensor hb2_tensor = {
        .data = (void *) s->hb2,
        .type = GGML_TYPE_F32,
        .ne = {hidden_dim, 1,1,1},
        .nb = {sizeof(float), sizeof(float)*hidden_dim, sizeof(float)*hidden_dim, sizeof(float)*hidden_dim}
    };

    for(unsigned long long l = 0; l < p->n_layers; l++) {

        rmsnorm((float *)a.data, x, (float *)tf->weights.lw[l].attn_norm->data, dim);
        // rmsnorm(OpTensor *xb , OpTensor *xb, OpTensor *weight);
        
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;
        k_tensor.data = s->k;
        v_tensor.data = s->v;

        // ggml_quantize_chunk(GGML_TYPE_Q8_0, w->wq + l * dim*dim, wq_q8, 0, dim, dim, NULL);
        ggml_compute_forward_op_mul_mat(&params, &q_tensor, tf->weights.lw[l].attn_q, &a);
        ggml_compute_forward_op_mul_mat(&params, &k_tensor, tf->weights.lw[l].attn_k, &a);
        ggml_compute_forward_op_mul_mat(&params, &v_tensor, tf->weights.lw[l].attn_v, &a);
        
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = (float *)q_tensor.data + h * head_size;
            // attention scores for this head
            // TODO: attention need OpTensor
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = (float *)a.data + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        ggml_compute_forward_op_mul_mat(&params, &a2, tf->weights.lw[l].attn_output, &a);
        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += ((float *)a2.data)[i];
        }

        rmsnorm((float *)a.data, x, (float *)w->lw[l].ffn_norm->data, dim);

        ggml_compute_forward_op_mul_mat(&params, &hb_tensor, w->lw[l].ffn_gate, &a);
        ggml_compute_forward_op_mul_mat(&params, &hb2_tensor, w->lw[l].ffn_up, &a);
    
        for (int i = 0; i < hidden_dim; i++) {
            float val = ((float *)hb_tensor.data)[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= ((float *)hb2_tensor.data)[i];
            ((float *)hb_tensor.data)[i] = val;
        }

        ggml_compute_forward_op_mul_mat(&params, &a, w->lw[l].ffn_down, &hb_tensor);
        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += ((float *)a.data)[i];
        }
    }
    rmsnorm(x, x, (float *)w->rms_att_weight->data, dim);
    struct OpTensor x_tensor = {
        .data = (void *) s->x,
        .type = GGML_TYPE_F32,
        .ne = {dim, 1,1,1},
        .nb = {sizeof(float), sizeof(float)*dim, sizeof(float)*dim, sizeof(float)*dim}
    };
    struct OpTensor logits_tensor = {
        .data = (void *) s->logits,
        .type = GGML_TYPE_F32,
        .ne = {p->vocab_size, 1,1,1},
        .nb = {sizeof(float), sizeof(float)*p->vocab_size, sizeof(float)*p->vocab_size, sizeof(float)*p->vocab_size}
    };
    ggml_compute_forward_op_mul_mat(&params, &logits_tensor, w->wcls, &x_tensor);

    free(w_data);

    return (float *)logits_tensor.data;
}

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}


void generate(Transformer *tf, smart::LlamaTokenizer *tk, Sampler *sampler, std::string prompt, int steps) {

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    auto prompt_tokens = tk->tokenize(prompt, true);
    num_prompt_tokens = prompt_tokens.size();
    
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
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
    printf("\n");

    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }
}
// -------------------------------
// debug info
void debug_meta_info(gguf_context *gguf_ctx, ggml_context *ggml_ctx);
void debug_tensors_info(gguf_context *gguf_ctx, ggml_context * ggml_ctx);
void debug_config_info(Config *c);
void debug_weight_info(std::string name, OpTensor *opt);
void debug_weights_info(TransformerWeights *w);

int main(int argc, char *argv[]) {
    Transformer transformer;
    build_transformer(&transformer, file_path);

    {
        // debug_meta_info(transformer.gguf_ctx, transformer.ggml_ctx);
        debug_tensors_info(transformer.gguf_ctx, transformer.ggml_ctx);
        debug_config_info(&transformer.config);
        debug_weights_info(&transformer.weights);
    }

    smart::LlamaTokenizer tokenizer(tokenizer_path);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    generate(&transformer, &tokenizer, &sampler, prompt, steps);

    free_sampler(&sampler);
    free_transformer(&transformer);

}

// --------------------------------
// debug info
void debug_meta_info(gguf_context *gguf_ctx, ggml_context *ggml_ctx) {
    {
        fmt::println("version     : {:10}", gguf_get_version(gguf_ctx));
        fmt::println("n_kv        : {:10}", gguf_get_n_kv(gguf_ctx));
        fmt::println("n_tensors   : {:10}", gguf_get_n_tensors(gguf_ctx));
        fmt::println("alignment   : {:10}", gguf_get_alignment(gguf_ctx));
        fmt::println("meta size   : {:10}", gguf_get_meta_size(gguf_ctx));
        fmt::println("data offset : {:10}", gguf_get_data_offset(gguf_ctx));
    }
    
    {
        for (auto i = 0; i < gguf_get_n_kv(gguf_ctx); i++) {
            auto key = gguf_get_key(gguf_ctx, i);
            auto v_type = gguf_get_kv_type(gguf_ctx, i);
            auto type_str = gguf_type_name(v_type);
            fmt::println("{:40}: {:4}", key, type_str);
        }
    }

    {
        for (auto i = 0; i < gguf_get_n_tensors(gguf_ctx); i++) {
            auto name = gguf_get_tensor_name(gguf_ctx, i);
            auto t_type = gguf_get_tensor_type(gguf_ctx, i);
            fmt::println("{:40}: {:6}: {:10}", name, ggml_type_name(t_type), gguf_get_tensor_offset(gguf_ctx, i));
        }
    }

    {
        fmt::println("GGML used mem        : {:10}", ggml_used_mem(ggml_ctx));
        fmt::println("GGML no alloc        : {:10}", ggml_get_no_alloc(ggml_ctx));
        fmt::println("GGML mem buffer      : {:10}", ggml_get_mem_buffer(ggml_ctx));
        fmt::println("GGML mem size        : {:10}", ggml_get_mem_size(ggml_ctx));
        fmt::println("GGML max tensor size : {:10}", ggml_get_max_tensor_size(ggml_ctx));
    }
}

void debug_tensors_info(gguf_context *gguf_ctx, ggml_context * ggml_ctx) {
    for (auto i = 0; i < gguf_get_n_tensors(gguf_ctx); i++) {
        auto t = ggml_get_tensor(ggml_ctx, gguf_get_tensor_name(gguf_ctx, i));
        
        fmt::println("{:40}|{:>5}|({:5},{:5},{:1},{:1})|{:10}|{:4}|{:4}|{:10}",
            ggml_get_name(t), 
            ggml_type_name(t->type), 
            t->ne[0], t->ne[1], t->ne[2], t->ne[3],
            ggml_get_data(t),
            ggml_type_size(t->type),
            ggml_blck_size(t->type),
            ggml_row_size(t->type, ggml_nelements(t))  // ne * ggml_type_size / ggml_blk_size (bytes)
        );
    }

}

void debug_config_info(Config *c) {
    fmt::println("dim       :{:6}", c->dim);
    fmt::println("hidden_dim:{:6}", c->hidden_dim);
    fmt::println("n_heads   :{:6}", c->n_heads);
    fmt::println("n_kv_heads:{:6}", c->n_kv_heads);
    fmt::println("n_layers  :{:6}", c->n_layers);
    fmt::println("seq_len   :{:6}", c->seq_len);
    fmt::println("vocab_size:{:6}", c->vocab_size);
}

void debug_weight_info(std::string name, OpTensor *opt) {
    auto out = fmt::format("{:5}|{:15}", ggml_type_name(opt->type), opt->data);
    fmt::println("{:15}: {}", name, out);
}

void debug_weights_info(TransformerWeights *w) {
    debug_weight_info("token embd", w->token_embedding_table);
    debug_weight_info("rms output", w->rms_att_weight);
    debug_weight_info("output", w->wcls);
    int layer = 0;
    for (auto &l: w->lw) {
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