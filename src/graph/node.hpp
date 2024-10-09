#pragma once

#include "common.hpp"
#include "ggml.h"
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <stdint.h>
#include <vector>
namespace smart {

enum class DataType {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
    Q4_0_4_4 = 31,
    Q4_0_4_8 = 32,
    Q4_0_8_8 = 33,
    TQ1_0   = 34,
    TQ2_0   = 35,

    TYPE_COUNT,
};

enum class OpType {
    OP_NONE = 0,
    
    OP_ADD,
    OP_MUL_MAT,
    OP_SIN,
    OP_COS,
    OP_SUM,
    OP_RMS_NORM,
    OP_SILU_HADAMARD,
    OP_ROPE,
    OP_SOFT_MAX,

    OP_MHA,
    OP_RES_CONN,

    OP_COUNT,
};


class Node {
public:
    std::vector<std::shared_ptr<Node>> prev; // inputs
    std::vector<std::shared_ptr<Node>> next; // outputs
    char node_type = 'n';

    Node() : node_type('n') {};
    virtual ~Node() = default;

};

inline void add_input(std::shared_ptr<Node> cur, std::shared_ptr<Node> in) {
    // TODO:
    SMART_ASSERT(cur && in);
    cur->prev.push_back(in);
    in->next.push_back(cur);
}

inline void add_output(std::shared_ptr<Node> cur, std::shared_ptr<Node> out) {
    // TODO:
    SMART_ASSERT(cur && out);
    cur->next.push_back(out);
    out->prev.push_back(cur);
}

// split tensor and op for multi-output
class Tensor : public Node {
public:

    DataType dtype;
    void *data; // others' data or container's data
    std::vector<char> container;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];

    Tensor() : dtype(DataType::F32), container(), data(), ne(1, 1, 1, 1), nb(1, 1, 1, 1) { node_type = 't';}
    Tensor(size_t size, std::initializer_list<int64_t> ne_, std::initializer_list<size_t> nb_, DataType dtype_ = DataType::F32) 
        : dtype(dtype_), 
          container(size)
    {
        node_type = 't';
        data = container.data();
        for (size_t i = 0; auto nei : ne_) {
            ne[i] = nei;
            i++;
        }
        for (size_t i = 0; auto nbi : nb_) {
            nb[i] = nbi;
            i++;
        }
    }

    Tensor(void* data_, std::initializer_list<int64_t> ne_, std::initializer_list<size_t> nb_, DataType dtype_ = DataType::F32)
        : dtype(dtype_),
          container(),
          data(data_)
        {
            node_type = 't';
            for (size_t i = 0; auto nei : ne_) {
                ne[i] = nei;
                i++;
            }
            for (size_t i = 0; auto nbi : nb_) {
                nb[i] = nbi;
                i++;
            }
        } 

    ~Tensor() {}

};
// tensors -> op -> tensors
class Operator : public Node {
public:
    // char node_type = 'o';

    OpType op_type;

    Operator() : op_type(OpType::OP_NONE) { node_type = 'o';}
    Operator(OpType op_type_) : op_type(op_type_) { node_type = 'o';}
    ~Operator() = default;
};

} // namespace smart