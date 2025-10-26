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



// 函数声明：我们向外界承诺，会提供这两个函数
void print_tensor(const Tensor& t);
// --- 新增代码 ---
// 声明一个函数，它将作为从CPU世界调用GPU功能的“桥梁”
void add_vectors_gpu(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c);
// --- 结束新增 ---

Tensor matmul_cpu(const Tensor& A, const Tensor& B);
Tensor softmax(const Tensor& input);
Tensor self_attention(const Tensor& Q, const Tensor& K, const Tensor& V);

void launch_add_one_kernel(float* d_data, int n);



// --- 【抄写块 1-B】---
Tensor matmul_cuda(const Tensor& A, const Tensor& B); // GPU版本
bool compare_tensors(const Tensor& a, const Tensor& b, float tolerance = 1e-5f); // 裁判函数

<<<<<<< HEAD

Tensor self_attention(const Tensor& Q, const Tensor& K, const Tensor& V);

Tensor self_attention_cuda_v2(const Tensor &Q,  const Tensor& K, const Tensor& V);

=======
>>>>>>> main
#endif // UTILS_H