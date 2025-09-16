#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <iostream>

// Tensor结构体，用来打包数据和形状
struct Tensor {
    std::vector<size_t> shape;
    std::vector<float> data;
};

// 函数声明：我们向外界承诺，会提供这两个函数
void print_tensor(const Tensor& t);
Tensor matmul(const Tensor& A, const Tensor& B);
Tensor softmax(const Tensor& input);
Tensor self_attention(const Tensor& Q, const Tensor& K, const Tensor& V);

#endif // UTILS_H