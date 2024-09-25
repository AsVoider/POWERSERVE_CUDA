#include "llama_tokenizer.hpp"
#include "CLI/CLI.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

// Globals
int GS = 0; // group size global for quantization of the weights

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

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

typedef struct {
    int8_t* q;    // quantized values
    float* s; // scaling factors
} QuantizedTensor;

typedef struct {
    // token embedding table
    QuantizedTensor *q_tokens; // (vocab_size, dim)
    float* token_embedding_table; // same, but dequantized

    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
    QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    QuantizedTensor *w1; // (layer, hidden_dim, dim)
    QuantizedTensor *w2; // (layer, dim, hidden_dim)
    QuantizedTensor *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    QuantizedTensor *wcls;
} TransformerWeights;

// typedef struct {
//     // current wave of activations
//     float *x; // activation at current time stamp (dim,)
//     float *xb; // same, but inside a residual branch (dim,)
//     float *xb2; // an additional buffer just for convenience (dim,)
//     float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
//     float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
//     float *q; // query (dim,)
//     float *k; // key (dim,)
//     float *v; // value (dim,)
//     float *att; // buffer for scores/attention values (n_heads, seq_len)
//     float *logits; // output logits
//     // kv cache
//     float* key_cache;   // (layer, seq_len, dim)
//     float* value_cache; // (layer, seq_len, dim)
// } RunState;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

// void malloc_run_state(RunState* s, Config* p) {
//     // we calloc instead of malloc to keep valgrind happy
//     int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
//     s->x = (float *)calloc(p->dim, sizeof(float));
//     s->xb = (float *)calloc(p->dim, sizeof(float));
//     s->xb2 = (float *)calloc(p->dim, sizeof(float));
//     s->hb = (float *)calloc(p->hidden_dim, sizeof(float));
//     s->hb2 = (float *)calloc(p->hidden_dim, sizeof(float));
//     s->q = (float *)calloc(p->dim, sizeof(float));
//     s->key_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
//     s->value_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
//     s->att = (float *)calloc(p->n_heads * p->seq_len, sizeof(float));
//     s->logits = (float *)calloc(p->vocab_size, sizeof(float));
//     // ensure all mallocs went fine
//     if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
//      || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
//         fprintf(stderr, "malloc failed!\n");
//         exit(EXIT_FAILURE);
//     }
// }

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = (float *) calloc(p->dim, sizeof(float));
    s->xb = (float *) calloc(p->dim, sizeof(float));
    s->xb2 = (float *) calloc(p->dim, sizeof(float));
    s->hb = (float *) calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float *) calloc(p->hidden_dim, sizeof(float));
    s->xq = (QuantizedTensor) { .q = (int8_t *) calloc(p->dim, sizeof(int8_t)), .s = (float *) calloc(p->dim, sizeof(float)) };
    s->hq = (QuantizedTensor) { .q = (int8_t *) calloc(p->hidden_dim, sizeof(int8_t)), .s = (float *) calloc(p->hidden_dim, sizeof(float)) };
    s->q = (float *) calloc(p->dim, sizeof(float));
    s->k = (float *) calloc(kv_dim, sizeof(float));
    s->v = (float *) calloc(kv_dim, sizeof(float));
    s->att = (float *) calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float *) calloc(p->vocab_size, sizeof(float));
    s->key_cache = (float *) calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float *) calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache
     || !s->value_cache) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

// void free_run_state(RunState* s) {
//     free(s->x);
//     free(s->xb);
//     free(s->xb2);
//     free(s->hb);
//     free(s->hb2);
//     free(s->q);
//     free(s->att);
//     free(s->logits);
//     free(s->key_cache);
//     free(s->value_cache);
// }

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {

        // find the max absolute value in the current group
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }
       // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
            qx->q[group * GS + i] = quantized;
        }
    }
}

/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    QuantizedTensor *res = (QuantizedTensor *) malloc(n * sizeof(QuantizedTensor));
    for(int i=0; i<n; i++) {
        /* map quantized int8 values*/
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        /* map scale factors */
        res[i].s = (float*)p;
        p = (float*)p + size_each / GS;
    }
    *ptr = p; // advance ptr to current position
    return res;
}

void memory_map_weights(TransformerWeights *w, Config* p, void* ptr, uint8_t shared_classifier) {
    int head_size = p->dim / p->n_heads;
    // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
    float* fptr = (float*) ptr; // cast our pointer to float*
    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;

    // now read all the quantized weights
    ptr = (void*)fptr; // now cast the pointer back to void*
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    // dequantize token embedding table
    w->token_embedding_table = (float *) malloc(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

// void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
//     int head_size = p->dim / p->n_heads;
//     // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
//     unsigned long long n_layers = p->n_layers;
//     w->token_embedding_table = ptr;
//     ptr += p->vocab_size * p->dim;
//     w->rms_att_weight = ptr;
//     ptr += n_layers * p->dim;
//     w->wq = ptr;
//     ptr += n_layers * p->dim * (p->n_heads * head_size);
//     w->wk = ptr;
//     ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
//     w->wv = ptr;
//     ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
//     w->wo = ptr;
//     ptr += n_layers * (p->n_heads * head_size) * p->dim;
//     w->rms_ffn_weight = ptr;
//     ptr += n_layers * p->dim;
//     w->w1 = ptr;
//     ptr += n_layers * p->dim * p->hidden_dim;
//     w->w2 = ptr;
//     ptr += n_layers * p->hidden_dim * p->dim;
//     w->w3 = ptr;
//     ptr += n_layers * p->dim * p->hidden_dim;
//     w->rms_final_weight = ptr;
//     ptr += p->dim;
//     ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
//     ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
//     w->wcls = shared_weights ? w->token_embedding_table : ptr;
// }

void read_checkpoint_quant(std::string checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint.c_str(), "rb");
    if (!file) { 
        fmt::println(stderr, "Couldn't open file {}", checkpoint);
        exit(EXIT_FAILURE); 
    }
    // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
    uint32_t magic_number;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (magic_number != 0x616b3432) { fprintf(stderr, "Bad magic number\n"); exit(EXIT_FAILURE); }
    // read in the version number (uint32), has to be 2
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (version != 2) { fprintf(stderr, "Bad version %d, need version 2\n", version); exit(EXIT_FAILURE); }
    int header_size = 256; // the header size for version 2 in bytes
    // read in the Config
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // read in flags
    uint8_t shared_classifier; // a byte to indicate if the classifier is shared
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    int group_size; // the group size used in quantization
    if (fread(&group_size, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    GS = group_size; // set as global, as it will be used in many places
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint.c_str(), O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = (float *) mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    void* weights_ptr = ((char*)*data) + header_size; // skip header bytes. char is 1 byte
    memory_map_weights(weights, config, weights_ptr, shared_classifier);
}


// void read_checkpoint(std::string checkpoint, Config* config, TransformerWeights* weights,
//                      int* fd, float** data, ssize_t* file_size) {
//     FILE *file = fopen(checkpoint.c_str(), "rb");
//     if (!file) { 
//         fmt::println(stderr, "Couldn't open file {}", checkpoint);
//         exit(EXIT_FAILURE); 
//     }
//     // read in the config header
//     if (fread(config, sizeof(Config), 1, file) != 1) { 
//         exit(EXIT_FAILURE); 
//     }
//     // negative vocab size is hacky way of signaling unshared weights. bit yikes.
//     int shared_weights = config->vocab_size > 0 ? 1 : 0;
//     config->vocab_size = abs(config->vocab_size);
//     // figure out the file size
//     fseek(file, 0, SEEK_END); // move file pointer to end of file
//     *file_size = ftell(file); // get the file size, in bytes
//     fclose(file);

//     // printf model configs
//     {
//         fmt::println("Config.dim        :{}", config->dim);
//         fmt::println("Config.hidden_dim :{}", config->hidden_dim);
//         fmt::println("Config.n_layers   :{}", config->n_layers);
//         fmt::println("Config.n_heads    :{}", config->n_heads);
//         fmt::println("Config.n_kv_heads :{}", config->n_kv_heads);
//         fmt::println("Config.vocab_size :{}", config->vocab_size);
//         fmt::println("Config.seq_len    :{}", config->seq_len);
//     }
    
//     // memory map the Transformer weights into the data pointer
//     *fd = open(checkpoint.c_str(), O_RDONLY); // open in read only mode
//     if (*fd == -1) { 
//         fprintf(stderr, "open failed!\n"); 
//         exit(EXIT_FAILURE); 
//     }
//     *data = (float *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
//     if (*data == MAP_FAILED) { 
//         fprintf(stderr, "mmap failed!\n"); 
//         exit(EXIT_FAILURE); 
//     }
//     float* weights_ptr = *data + sizeof(Config)/sizeof(float);
//     memory_map_weights(weights, config, weights_ptr, shared_weights);
// }

void build_transformer(Transformer *t, std::string checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    // read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    read_checkpoint_quant(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // free QuantizedTensors
    free(t->weights.q_tokens);
    free(t->weights.token_embedding_table);
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    if(t->weights.wcls != t->weights.q_tokens) { free(t->weights.wcls); }
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// void free_transformer(Transformer* t) {
//     // close the memory mapping
//     if (t->data != MAP_FAILED) { 
//         munmap(t->data, t->file_size); 
//     }
//     if (t->fd != -1) { 
//         close(t->fd); 
//     }
//     // free the RunState buffers
//     free_run_state(&t->state);
// }

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


void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized

    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {

        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;

        // do the matmul in groups of GS
        int j;
        for (j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }

        xout[i] = val;
    }
}

// void matmul(float* xout, float* x, float* w, int n, int d) {
//     // W (d,n) @ x (n,) -> xout (d,)
//     // by far the most amount of time is spent inside this little function
//     int i;
//     #pragma omp parallel for private(i)
//     for (i = 0; i < d; i++) {
//         float val = 0.0f;
//         for (int j = 0; j < n; j++) {
//             val += w[i * n + j] * x[j];
//         }
//         xout[i] = val;
//     }
// }

float* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
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

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        // int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        // s->k = s->key_cache + loff + pos * kv_dim;
        // s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, dim);
        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);
        // matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        // matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        // matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

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

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

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
        quantize(&s->xq, s->xb, dim);
        matmul(s->xb2, &s->xq, w->wo + l, dim, dim);
        // matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);
        // matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        // matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

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
        quantize(&s->hq, s->hb, hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);
        // matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    // matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void generate(Transformer *transformer, smart::LlamaTokenizer *tokenizer, Sampler *sampler, std::string prompt, int steps) {

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    auto prompt_tokens = tokenizer->tokenize(prompt, true);
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
        float* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { 
            break; 
        }

        // print the token as string, decode it with the Tokenizer object
        auto piece = tokenizer->to_string(next);
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

int main(int argc, char *argv[]) {
    // default parameters
    std::string checkpoint_path = "/home/feiychen/桌面/llama2-7b_Q8_0.bin"; // "/home/zwb/Downloads/Llama-2-7b-chat-hf/llama2_7b.bin";  // e.g. out/model.bin
    std::string tokenizer_path = "/home/feiychen/桌面/llama2_7b_vocab.gguf";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 64;            // number of steps to run for
    std::string prompt = "One day,";        // prompt string
    unsigned long long rng_seed = 2024923; // seed rng with time by default

    // print arguments info
    {
        fmt::println("checkpoint_path: {}", checkpoint_path);
        fmt::println("tokenizer_path : {}", tokenizer_path);
        fmt::println("temperature    : {}", temperature);
        fmt::println("topp           : {}", topp);
        fmt::println("steps          : {}", steps);
        fmt::println("prompt         : {}", prompt);
        fmt::println("rng_seed       : {}", rng_seed);
    }

    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);

    smart::LlamaTokenizer tokenizer(tokenizer_path);

    {
        fmt::println("#vocab   : {}", tokenizer.n_vocabs());
        fmt::println("BOS token: {}", tokenizer.bos_token());
    }

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    generate(&transformer, &tokenizer, &sampler, prompt, steps);

    free_sampler(&sampler);
    free_transformer(&transformer);
    return 0;
}