#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <cassert>
#include <stdbool.h>
#include <stddef.h>
#include <string>
#include <vector>
#include <ggml.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "fmt/base.h"
#include "gguf.hpp"
#include "llama_tokenizer.hpp"

// -------------------
using namespace smart;

// -------------------
ssize_t file_size = 0;
std::string file_path = "/home/zwb/SS/models/Llama-2-7b-chat-hf/llama-2-7b.f32.gguf";
std::string tokenizer_path = "/home/zwb/Downloads/Llama-2-7b-chat-hf/llama2_7b_vocab.gguf";
GGUFContext ctx = {};
float temperature = 1.0f;
float topp = 0.9f;
unsigned long long rng_seed = 2024923;
std::string prompt = "One day,";
int steps = 64;

// -------------------
// common meta data idx
ssize_t max_seq_len_idx      = -1; // "llama.context_length"
ssize_t dim_idx              = -1; // "llama.embedding_length"
ssize_t hidden_dim_idx       = -1; // "llama.feed_forward_length"
ssize_t n_layers_idx         = -1; // "llama.block_count"
ssize_t n_heads_idx          = -1; // "llama.attention.head_count"
ssize_t n_kv_heads_idx       = -1; // "llama.attention.head_count_kv"
ssize_t rope_dim_count_idx   = -1; // "llama.rope.dimension_count"
ssize_t align_idx            = -1; // "general.alignment"
ssize_t vocab_size_idx       = -1; // "llama.vocab_size"

// 291 layers = 32 * 9 + 1 + 1 + 1
struct LayerWeights {
    // weights for rmsnorms
    GGUFTensorInfo *attn_norm;  // (4096, )
    GGUFTensorInfo *ffn_norm;   // (4096, )

    // weights for matmuls. not dim == n_heads * head_size
    GGUFTensorInfo *attn_q;      // (4096, 4096)
    GGUFTensorInfo *attn_k;      // (4096, 4096)
    GGUFTensorInfo *attn_v;      // (4096, 4096)
    GGUFTensorInfo *attn_output; // (4096, 4096)

    GGUFTensorInfo *ffn_gate;  // (4096, 11008)
    GGUFTensorInfo *ffn_up;    // (4096, 11008)
    GGUFTensorInfo *ffn_down;  // (11008, 4096)
};

std::vector<LayerWeights> weights;
GGUFTensorInfo * token_embd_weight = nullptr;  // (4096, 32000)
GGUFTensorInfo * output_norm_weight = nullptr;      // (4096, )
GGUFTensorInfo * output_weight = nullptr;      // (4096, 32000)

struct TransformerWeights_F32 {
    char *w_ptr; // weight data start pointer
    // token embedding table
    float *token_embedding_table; // (4096, 32000)
    // weights for rmsnorms
    float *rms_att_weight;  // (layer, 4096, )
    float *rms_ffn_weight;   // (layer, 4096, )

    // weights for matmuls. not dim == n_heads * head_size
    float *wq;      // (layer, 4096, 4096)
    float *wk;      // (layer, 4096, 4096)
    float *wv;      // (layer, 4096, 4096)
    float *wo;      // (layer, 4096, 4096)

    float *w1;  // gate (layer, 4096, 11008)
    float *w2;  // down (layer, 11008, 4096)
    float *w3;  // up   (layer, 4096, 11008)

    // final rmsnorm
    float* rms_final_weight; // (4096,) -> output_norm_weight
    // (optional) classifier weights for the logits, on the last layer
    float* wcls; // -> final_weight
};

TransformerWeights_F32 f32_w;
std::vector<GGUFTensorInfo *> order_tensor_info; // trace tensor info in head's order

// -----------------
// llama2.c
typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

Config config;

typedef struct {
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
} RunState;

RunState state;

// -------------------
// read file contents

template<typename T>
bool read_item(std::ifstream& file, T& v) {
    return !!file.read(reinterpret_cast<char *>(&v), sizeof(v));
}

bool read_item(std::ifstream& file, std::string& v) {
    int64_t len = -1;
    if (!read_item(file, len)) {
        return false;
    }
    v.resize(len, '\0');
    return !!file.read(v.data(), len);
}

std::string read_string(std::ifstream& file) {
    std::string res;
    read_item(file, res);
    return res;
}

int64_t read_integer(std::ifstream& file, GGUFValueType vt, int64_t default_value=-1) {
    int64_t res = default_value;
    switch (vt) {
        case GGUFValueType::UINT8:
        case GGUFValueType::UINT16:
        case GGUFValueType::UINT32:
        case GGUFValueType::UINT64:
        case GGUFValueType::INT64:
            file.read(reinterpret_cast<char *>(&res), s_vtype_descriptors[int(vt)].size);
            break;
        case GGUFValueType::INT8:
            if (int8_t v; file.read(reinterpret_cast<char*>(&v), s_vtype_descriptors[int(vt)].size)) {
                res = v;
            }
            break;
        case GGUFValueType::INT16:
            if (int16_t v; file.read(reinterpret_cast<char*>(&v), s_vtype_descriptors[int(vt)].size)) {
                res = v;
            }
            break;
        case GGUFValueType::INT32:
            if (int32_t v; file.read(reinterpret_cast<char*>(&v), s_vtype_descriptors[int(vt)].size)) {
                res = v;
            }
            break;
        case GGUFValueType::BOOL:
            if (bool v; file.read(reinterpret_cast<char*>(&v), s_vtype_descriptors[int(vt)].size)) {
                res = v;
            }
            break;
        default:
            break;
    }
    return res;
}

double read_float(std::ifstream& file, GGUFValueType vt, double default_value=0.) {
    if (vt == GGUFValueType::FLOAT32) {
        float v = 0.;
        if (!file.read(reinterpret_cast<char*>(&v), sizeof(v))) {
            return default_value;
        }
        return v;
    } 
    else if (vt == GGUFValueType::FLOAT64) {
        double v = 0.;
        if (!file.read(reinterpret_cast<char*>(&v), sizeof(v))) {
            return default_value;
        }
        return v;
    } else {
        return default_value;
    }
}

bool read_array_str(std::ifstream& file, GGUFArray &arr) {
    for (int64_t i = 0; i < arr.n; i++) {
        // ((std::string *)arr.data)[i] = read_string(file);
        auto s = read_string(file);
        ((std::string *)arr.data)[i] = s;
    }
    return true;
}

bool read_arr(std::ifstream& file, GGUFArray& arr) {
    GGUFValueType vt = static_cast<GGUFValueType>(read_integer(file, GGUFValueType::INT32));
    arr.type = GGUFType(vt);
    int64_t size = read_integer(file, GGUFValueType::INT64, -1);
    assert(size >= 1 && size <= (int64_t(1) << 40));
    arr.n = size;
    
    // fmt::println("n: {}, type: {} size: {}", arr.n, GGUFType2Str(arr.type), s_vtype_descriptors[int(arr.type)].size);
    if (is_string(GGUFValueType(arr.type))) {
        arr.data = calloc(arr.n, sizeof(std::string));
        return read_array_str(file, arr);
    } else if (is_array(GGUFValueType(arr.type))) {
        fmt::println("{} Error type for array", __func__);
        return false;
    } else {
        arr.data = calloc(arr.n, s_vtype_descriptors[int(arr.type)].size);
        memset(arr.data, 'a', arr.n * s_vtype_descriptors[int(arr.type)].size);
        for (auto i = 0; i < arr.n; i++) {
            switch (arr.type) {
                case GGUFType::GGUF_TYPE_FLOAT32: ((float *) arr.data)[i] = read_float(file, GGUFValueType(arr.type)); break;
                case GGUFType::GGUF_TYPE_FLOAT64: ((double *) arr.data)[i] = read_float(file, GGUFValueType(arr.type)); break;
                case GGUFType::GGUF_TYPE_INT8   : ((int8_t *) arr.data)[i] = read_integer(file, GGUFValueType(arr.type)); break;
                case GGUFType::GGUF_TYPE_UINT8  : ((uint8_t *) arr.data)[i] = read_integer(file, GGUFValueType(arr.type)); break;
                case GGUFType::GGUF_TYPE_INT16  : ((int16_t *) arr.data)[i] = read_integer(file, GGUFValueType(arr.type)); break;
                case GGUFType::GGUF_TYPE_UINT16 : ((uint16_t *) arr.data)[i] = read_integer(file, GGUFValueType(arr.type)); break;
                case GGUFType::GGUF_TYPE_INT32  : ((int32_t *) arr.data)[i] = read_integer(file, GGUFValueType(arr.type)); break;
                case GGUFType::GGUF_TYPE_UINT32 : ((uint32_t *) arr.data)[i] = read_integer(file, GGUFValueType(arr.type)); break;
                case GGUFType::GGUF_TYPE_INT64  : ((int64_t *) arr.data)[i] = read_integer(file, GGUFValueType(arr.type)); break;
                case GGUFType::GGUF_TYPE_UINT64 : ((uint64_t *) arr.data)[i] = read_integer(file, GGUFValueType(arr.type)); break;
                case GGUFType::GGUF_TYPE_BOOL   : ((bool *) arr.data)[i] = read_integer(file, GGUFValueType(arr.type)); break;
                default: return false;
            }
        }
        return true;
        // return !!file.read(reinterpret_cast<char*>(arr.data), s_vtype_descriptors[int(arr.type)].size * arr.n);
    }
    
}

// ------------------
// weights

void mapping_weights() {
    ctx.fd = open(file_path.c_str(), O_RDONLY);
    if (ctx.fd == -1) { 
        fprintf(stderr, "open failed!\n"); 
        exit(EXIT_FAILURE); 
    }
    ctx.data = (char *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, ctx.fd, 0);
    if (ctx.data == MAP_FAILED) { 
        fprintf(stderr, "mmap failed!\n"); 
        exit(EXIT_FAILURE); 
    }
    f32_w.w_ptr = ctx.data + ctx.offset;

    // TODO: memcpy as char OOM -> mmap
    {
        f32_w.token_embedding_table = (float *)(f32_w.w_ptr + token_embd_weight->offset);
        f32_w.wcls = (float *)(f32_w.w_ptr + output_weight->offset);
        f32_w.rms_final_weight = (float *)(f32_w.w_ptr + output_norm_weight->offset);
    }
}

void unmapping_weights() {
    munmap(ctx.data, file_size);
    close(ctx.fd);
}
// ------------------
// llama2.c

void malloc_run_state() {
    // we calloc instead of malloc to keep valgrind happy
    auto &p = config;
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

void free_run_state() {
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

Sampler sampler;

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

int sample(float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    next = sample_argmax(logits, sampler.vocab_size);
    return next;
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

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

// int sample(float* logits) {
//     // sample the token given the logits and some hyperparameters
//     int next;
//     if (sampler.temperature == 0.0f) {
//         // greedy argmax sampling: take the token with the highest probability
//         next = sample_argmax(logits, sampler.vocab_size);
//     } else {
//         // apply the temperature to the logits
//         for (int q=0; q<sampler.vocab_size; q++) { 
//             logits[q] /= sampler.temperature; 
//         }
//         // apply softmax to the logits to get the probabilities for next token
//         softmax(logits, sampler.vocab_size);
//         // flip a (float) coin (this is our source of entropy for sampling)
//         float coin = random_f32(&sampler.rng_state);
//         // we sample from this distribution to get the next token
//         if (sampler.topp <= 0 || sampler.topp >= 1) {
//             // simply sample from the predicted probability distribution
//             next = sample_mult(logits, sampler.vocab_size, coin);
//         } else {
//             // top-p (nucleus) sampling, clamping the least likely tokens to zero
//             next = sample_topp(logits, sampler.vocab_size, sampler.topp, sampler.probindex, coin);
//         }
//     }
//     return next;
// }

void build_sampler(int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler.vocab_size = vocab_size;
    sampler.temperature = temperature;
    sampler.topp = topp;
    sampler.rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler.probindex = (ProbIndex *)malloc(sampler.vocab_size * sizeof(ProbIndex));
}

void free_sampler() {
    free(sampler.probindex);
}

float* forward(int token, int pos) {

    // a few convenience variables
    Config* p = &config;
    TransformerWeights_F32* w = &f32_w;
    RunState* s = &state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        w->rms_att_weight = (float *)(w->w_ptr + weights[l].attn_norm->offset);
        w->wq = (float *)(w->w_ptr + weights[l].attn_q->offset);
        w->wk = (float *)(w->w_ptr + weights[l].attn_k->offset);
        w->wv = (float *)(w->w_ptr + weights[l].attn_v->offset);
        w->wo = (float *)(w->w_ptr + weights[l].attn_output->offset);
        w->rms_ffn_weight = (float *)(w->w_ptr + weights[l].ffn_norm->offset);
        w->w1 = (float *)(w->w_ptr + weights[l].ffn_gate->offset);
        w->w3 = (float *)(w->w_ptr + weights[l].ffn_up->offset);
        w->w2 = (float *)(w->w_ptr + weights[l].ffn_down->offset);

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq, dim, dim);
        matmul(s->k, s->xb, w->wk, dim, kv_dim);
        matmul(s->v, s->xb, w->wv, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
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

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
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
            float* xb = s->xb + h * head_size;
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

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void generate(smart::LlamaTokenizer &tokenizer) {

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    auto prompt_tokens = tokenizer.tokenize(prompt, true);
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
        float* logits = forward(token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { 
            break; 
        }

        // print the token as string, decode it with the Tokenizer object
        auto piece = tokenizer.to_string(next);
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


// ------------------
// tools

// split string by char
void string_split(const std::string& str, const char split, std::vector<std::string>& res)
{
	if (str == "")		
        return;
	
    // 在字符串末尾也加入分隔符，方便截取最后一段
    std::string strs = str + split;
	size_t pos = strs.find(split);
 
	// 若找不到内容则字符串搜索函数返回 npos
	while (pos != strs.npos)
	{
		std::string temp = strs.substr(0, pos);
		res.push_back(temp);
		// 去掉已分割的字符串,在剩下的字符串中进行分割
		strs = strs.substr(pos + 1, strs.size());
		pos = strs.find(split);
	}
}

// ------------------

int main(int argc, char *argv[]) {

    if (argc == 2)
        file_path = argv[1];

    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    assert(file.is_open());
    file_size = file.tellg();
    file.seekg(std::ios::beg);

    // get header
    {
        GGUFHeader &fh = ctx.header;
        assert(file.read(reinterpret_cast<char *>(&fh), sizeof(fh)));
        assert(fh.magic_number == GGUFFILETAG);
        // fmt::println("magic number : {0:#10x}", fh.magic_number);
        // fmt::println("version      : {:10}", fh.version);
        // fmt::println("tensor count : {:10}", fh.tensor_count);
        // fmt::println("kv data count: {:10}", fh.metadata_kv_count);
    }

    // read the kv pairs
    {
        const uint64_t n_kv = ctx.header.metadata_kv_count;

        // header.n_kv will hold the actual value of pairs that were successfully read in the loop below
        ctx.header.metadata_kv_count = 0;
        ctx.kv = (GGUFKV *)calloc(n_kv, sizeof(GGUFKV));
        for (uint64_t i = 0; i < n_kv; ++i) {
            GGUFKV * cur_kv = &ctx.kv[i];
            // string for key
            auto k = read_string(file);
            cur_kv->key = k;
            // mark common key-value pairs' index
            {
                if (k == "llama.context_length") { max_seq_len_idx = i; } 
                else if (k == "llama.embedding_length") { dim_idx = i; }
                else if (k == "llama.feed_forward_length") { hidden_dim_idx = i; }
                else if (k == "llama.block_count") { n_layers_idx = i; }
                else if (k == "llama.attention.head_count") { n_heads_idx = i; }
                else if (k == "llama.attention.head_count_kv") { n_kv_heads_idx = i; }
                else if (k == "llama.rope.dimension_count") { rope_dim_count_idx = i; }
                else if (k == "general.alignment") { align_idx = i; }
                else if (k == "llama.vocab_size") { vocab_size_idx = i; }
            }

            // int32 for value type
            auto t = GGUFType(read_integer(file, GGUFValueType::INT32));
            cur_kv->type = t;
            
            std::string s;
            int64_t z;
            double f;
            GGUFArray ar;
            switch (cur_kv->type) {
                case GGUFType::GGUF_TYPE_STRING: 
                    s = read_string(file); 
                    cur_kv->value.str = s; 
                    break;
                // case GGUFType::GGUF_TYPE_INT8:
                //     z = read_integer(file, GGUFValueType(cur_kv->type)); 
                //     cur_kv->value.int8 = z;
                //     break;
                // case GGUFType::GGUF_TYPE_UINT8:
                //     z = read_integer(file, GGUFValueType(cur_kv->type)); 
                //     cur_kv->value.uint8 = z;
                //     break;
                // case GGUFType::GGUF_TYPE_INT16:
                //     z = read_integer(file, GGUFValueType(cur_kv->type)); 
                //     cur_kv->value.int16 = z;
                //     break;
                // case GGUFType::GGUF_TYPE_UINT16:
                //     z = read_integer(file, GGUFValueType(cur_kv->type)); 
                //     cur_kv->value.uint16 = z;
                //     break;
                // case GGUFType::GGUF_TYPE_INT32:
                //     z = read_integer(file, GGUFValueType(cur_kv->type)); 
                //     cur_kv->value.int32 = z;
                //     break;
                // case GGUFType::GGUF_TYPE_UINT32:
                //     z = read_integer(file, GGUFValueType(cur_kv->type)); 
                //     cur_kv->value.uint32 = z;
                //     break;
                // case GGUFType::GGUF_TYPE_INT64:
                //     z = read_integer(file, GGUFValueType(cur_kv->type)); 
                //     cur_kv->value.int64 = z;
                //     break;
                // case GGUFType::GGUF_TYPE_UINT64:
                //     z = read_integer(file, GGUFValueType(cur_kv->type)); 
                //     cur_kv->value.uint64 = z;
                //     break;
                // case GGUFType::GGUF_TYPE_BOOL:
                //     z = read_integer(file, GGUFValueType(cur_kv->type)); 
                //     cur_kv->value.bool_ = z;
                //     break;
                case GGUFType::GGUF_TYPE_INT8:
                case GGUFType::GGUF_TYPE_UINT8:
                case GGUFType::GGUF_TYPE_INT16:
                case GGUFType::GGUF_TYPE_UINT16:
                case GGUFType::GGUF_TYPE_INT32:
                case GGUFType::GGUF_TYPE_UINT32:
                case GGUFType::GGUF_TYPE_INT64:
                case GGUFType::GGUF_TYPE_UINT64:
                case GGUFType::GGUF_TYPE_BOOL:
                    z = read_integer(file, GGUFValueType(cur_kv->type)); 
                    cur_kv->value.uint64 = z;
                    break;
                // case GGUFType::GGUF_TYPE_FLOAT32:
                //     f = read_float(file, GGUFValueType(cur_kv->type));
                //     cur_kv->value.float32 = f;
                //     break;
                // case GGUFType::GGUF_TYPE_FLOAT64:
                //     f = read_float(file, GGUFValueType(cur_kv->type));
                //     cur_kv->value.float64 = f;
                //     break;
                case GGUFType::GGUF_TYPE_FLOAT32:
                case GGUFType::GGUF_TYPE_FLOAT64:
                    f = read_float(file, GGUFValueType(cur_kv->type));
                    cur_kv->value.float64 = f;
                    break;
                case GGUFType::GGUF_TYPE_ARRAY:
                    assert(read_arr(file, cur_kv->value.arr));
                    break;
                default: 
                    break;
            }
            
            // fmt::println("{:40} : [{:>7}] {:10}", cur_kv->key.data(), GGUFType2Str(cur_kv->type), GGUFValue2Str(cur_kv->value, cur_kv->type));
        }

        {
            // fmt::println("[{:2}]: {:30}: {}", max_seq_len_idx, ctx.kv[max_seq_len_idx].key, GGUFValue2Str(ctx.kv[max_seq_len_idx].value, ctx.kv[max_seq_len_idx].type));
            // fmt::println("[{:2}]: {:30}: {}", dim_idx, ctx.kv[dim_idx].key, GGUFValue2Str(ctx.kv[dim_idx].value, ctx.kv[dim_idx].type));
            // fmt::println("[{:2}]: {:30}: {}", hidden_dim_idx, ctx.kv[hidden_dim_idx].key, GGUFValue2Str(ctx.kv[hidden_dim_idx].value, ctx.kv[hidden_dim_idx].type));
            // fmt::println("[{:2}]: {:30}: {}", n_layers_idx, ctx.kv[n_layers_idx].key, GGUFValue2Str(ctx.kv[n_layers_idx].value, ctx.kv[n_layers_idx].type));
            // fmt::println("[{:2}]: {:30}: {}", n_heads_idx, ctx.kv[n_heads_idx].key, GGUFValue2Str(ctx.kv[n_heads_idx].value, ctx.kv[n_heads_idx].type));
            // fmt::println("[{:2}]: {:30}: {}", n_kv_heads_idx, ctx.kv[n_kv_heads_idx].key, GGUFValue2Str(ctx.kv[n_kv_heads_idx].value, ctx.kv[n_kv_heads_idx].type));
            // fmt::println("[{:2}]: {:30}: {}", rope_dim_count_idx, ctx.kv[rope_dim_count_idx].key, GGUFValue2Str(ctx.kv[rope_dim_count_idx].value, ctx.kv[rope_dim_count_idx].type));

            config.dim = ctx.kv[dim_idx].value.uint32;
            config.hidden_dim = ctx.kv[hidden_dim_idx].value.uint32;
            config.n_layers = ctx.kv[n_layers_idx].value.uint32;
            config.n_heads = ctx.kv[n_heads_idx].value.uint32;
            config.n_kv_heads = ctx.kv[n_kv_heads_idx].value.uint32;
            config.vocab_size = ctx.kv[vocab_size_idx].value.uint32;
            config.seq_len = ctx.kv[max_seq_len_idx].value.uint32;
        }
    }

    // for convenience
    {
        weights.resize(ctx.kv[n_layers_idx].value.uint32);
    }

    // read the tensor infos
    {
        auto &tensor_infos = ctx.infos;
        for (uint64_t i = 0; i < ctx.header.tensor_count; i++) {
            auto name = read_string(file);
            tensor_infos[name] = GGUFTensorInfo();
            
            GGUFTensorInfo *cur_info = &tensor_infos[name];
            cur_info->name = name;

            for (int j = 0; j < GGMLMAXDIMS; j++) {
                cur_info->ne[j] = 1;
            }

            cur_info->n_dims = read_integer(file, GGUFValueType::INT32);

            for (auto j = 0; j < cur_info->n_dims; j++) {
                cur_info->ne[j] = read_integer(file, GGUFValueType::INT64);
            }

            cur_info->type = GGMLType(read_integer(file, GGUFValueType::INT32));
            cur_info->offset = read_integer(file, GGUFValueType::INT64);

            gguf_tensor_info_sanitize(cur_info);

            // TODO: make sure there is no duplicated tensor names
            
            // fmt::println("[{:3}] {}", i, GGUFTensorInfo2Str(tensor_infos[name]));

            // for convenience
            {
                order_tensor_info.push_back(&tensor_infos[name]);
                if (name == "token_embd.weight") { token_embd_weight = &tensor_infos[name]; } 
                else if (name == "output_norm.weight") { output_norm_weight = &tensor_infos[name]; } 
                else if (name == "output.weight"){ output_weight = &tensor_infos[name]; } 
                else {
                    std::vector<std::string> toks;
                    string_split(name, '.', toks);
                    auto layer = std::stoi(toks[1]);
                    auto tensor_name = toks[2];

                    if (tensor_name == "attn_k") { weights[layer].attn_k = &tensor_infos[name];}
                    else if (tensor_name == "attn_q") { weights[layer].attn_q = &tensor_infos[name];}
                    else if (tensor_name == "attn_v") { weights[layer].attn_v = &tensor_infos[name];}
                    else if (tensor_name == "attn_output") { weights[layer].attn_output = &tensor_infos[name];}
                    else if (tensor_name == "attn_norm") { weights[layer].attn_norm = &tensor_infos[name];}
                    else if (tensor_name == "ffn_norm") { weights[layer].ffn_norm = &tensor_infos[name];}
                    else if (tensor_name == "ffn_up") { weights[layer].ffn_up = &tensor_infos[name];}
                    else if (tensor_name == "ffn_down") { weights[layer].ffn_down = &tensor_infos[name];}
                    else if (tensor_name == "ffn_gate") { weights[layer].ffn_gate = &tensor_infos[name];}
                }
            }
        }
    }

    ctx.alignment = GGUFDEFAULTALIGNMENT;
    if (align_idx >= 0)
        ctx.alignment = ctx.kv[align_idx].value.uint32;

    // we require the data section to be aligned, so take into account any padding
    size_t file_pos = size_t(file.tellg());
    file_pos = (file_pos + ctx.alignment - 1) & ~size_t(ctx.alignment - 1);
    // store the current file offset - this is where the data section starts
    ctx.offset = file_pos;
    // fmt::println("context offset: {}", ctx.offset);
    
    // compute the total size of the data section, taking into account the alignment
    {
        ctx.size = 0;
        for(auto ti: order_tensor_info) {
            size_t offset = file_pos + ti->offset;
            size_t num_items = 1;
            int64_t ne = 1;
            for (int i = 0; i < ti->n_dims; ++i) {
                ne *= ti->ne[i];
            }

            // TODO: not use ggml.h
            assert(ggml_blck_size((ggml_type)ti->type) != 0);
            assert(ne % ggml_blck_size((ggml_type)ti->type) == 0);

            const size_t size_cur = ggml_row_size((ggml_type)ti->type, ne);
            ctx.size += GGML_PAD(size_cur, ctx.alignment);
        }
        // fmt::println("padding ctx size: {}", ctx.size);
    }

    // load the tensor data
    {
        // alloc_weights();
        file.close();
        mapping_weights();
        malloc_run_state();
        smart::LlamaTokenizer tokenizer(tokenizer_path);
        build_sampler(config.vocab_size, temperature, topp, rng_seed);
        generate(tokenizer);

        {
            // fmt::println("WQ");
            // for (auto j = 0; j < config.n_layers; j++) {
            //     f32_w.wq = (float *)(f32_w.w_ptr + weights[j].attn_q->offset);
            //     for (auto i = 0; i < config.dim * config.dim; i++) {
            //         fmt::println("{}: {}", i, f32_w.wq[i]);
            //     }
            // }
            // fmt::println("WK");
            // for (auto j = 0; j < config.n_layers; j++) {
            //     f32_w.wk = (float *)(f32_w.w_ptr + weights[j].attn_k->offset);
            //     for (auto i = 0; i < config.dim * config.dim; i++) {
            //         fmt::println("{}: {}", i, f32_w.wk[i]);
            //     }
            // }
            // fmt::println("WV");
            // for (auto j = 0; j < config.n_layers; j++) {
            //     f32_w.wv = (float *)(f32_w.w_ptr + weights[j].attn_v->offset);
            //     for (auto i = 0; i < config.dim * config.dim; i++) {
            //         fmt::println("{}: {}", i, f32_w.wv[i]);
            //     }
            // }
            // fmt::println("WO");
            // for (auto j = 0; j < config.n_layers; j++) {
            //     f32_w.wo = (float *)(f32_w.w_ptr + weights[j].attn_output->offset);
            //     for (auto i = 0; i < config.dim * config.dim; i++) {
            //         fmt::println("{}: {}", i, f32_w.wo[i]);
            //     }
            // }

            // fmt::println("W1");
            // for (auto j = 0; j < config.n_layers; j++) {
            //     f32_w.w1 = (float *)(f32_w.w_ptr + weights[j].ffn_gate->offset);
            //     for (auto i = 0; i < config.hidden_dim * config.dim; i++) {
            //         fmt::println("{}: {}", i, f32_w.w1[i]);
            //     }
            // }
            // fmt::println("W2");
            // for (auto j = 0; j < config.n_layers; j++) {
            //     f32_w.w2 = (float *)(f32_w.w_ptr + weights[j].ffn_down->offset);
            //     for (auto i = 0; i < config.hidden_dim * config.dim; i++) {
            //         fmt::println("{}: {}", i, f32_w.w2[i]);
            //     }
            // }
            // fmt::println("W3");
            // for (auto j = 0; j < config.n_layers; j++) {
            //     f32_w.w3 = (float *)(f32_w.w_ptr + weights[j].ffn_up->offset);
            //     for (auto i = 0; i < config.hidden_dim * config.dim; i++) {
            //         fmt::println("{}: {}", i, f32_w.w3[i]);
            //     }
            // }
            // fmt::println("RMS FINAL WEIGHT");
            // for (auto i = 0; i < config.dim; i++) {
            //     fmt::println("{}: {}", i, f32_w.rms_final_weight[i]);
            // }
            // fmt::println("RMS Weights");
            // for (auto i = 0; i < config.vocab_size * config.dim; i++)
            //     fmt::println("{}: {}", i, f32_w.final_weight[i]);
            // fmt::println("embeding");
            // for (auto i = 0; i < config.vocab_size * config.dim; i++)
            //     fmt::println("{}: {}", i, f32_w.token_embedding_table[i]);
        }

        free_sampler();
        free_run_state();
        unmapping_weights();
        // free_weights();
    }

    return 0;
}