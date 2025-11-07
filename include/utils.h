#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <iostream>

// Tensor结构体，用来打包数据和形状
struct Tensor {
    size_t rows;
    size_t cols;
    std::vector<float> data;
//conveniently creat 
    Tensor(size_t r =0,size_t c =0) :rows(r) , cols(c) ,data(r*c,0.0f){};
};


#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdio> // 为了 fprintf

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error("CUDA error"); \
        } \
    } while (0)
// ------------------------------------



void print_tensor(const Tensor& t);
void add_vectors_gpu(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c);

Tensor matmul_cpu(const Tensor& A, const Tensor& B);
Tensor softmax(const Tensor& input);
Tensor self_attention(const Tensor& Q, const Tensor& K, const Tensor& V);

void launch_add_one_kernel(float* d_data, int n);


Tensor matmul_cuda(const Tensor& A, const Tensor& B); // GPU版本
bool compare_tensors(const Tensor& a, const Tensor& b, float tolerance = 1e-5f); // 裁判函数


Tensor self_attention(const Tensor& Q, const Tensor& K, const Tensor& V);

Tensor self_attention_cuda_v2(const Tensor &Q,  const Tensor& K, const Tensor& V);

Tensor layernorm_cpu(const Tensor &input , const Tensor &gamma, const Tensor &beta);
Tensor layernorm_cuda(const Tensor &input , const Tensor &gamma, const Tensor &beta);

Tensor gelu_cpu(const Tensor &input );
Tensor gelu_cuda(const Tensor &input );

Tensor add_bias_cpu(const Tensor & hidden,const Tensor & bias);
Tensor add_bias_cuda(const Tensor & hidden,const Tensor & bias);

Tensor ffn_cpu(const Tensor & input,const Tensor & w1,const Tensor & b1,const Tensor & w2,const Tensor  & b2);
Tensor ffn_cuda(const Tensor & input,const Tensor & w1,const Tensor & b1,const Tensor & w2,const Tensor & b2);
#endif // UTILS_H