#include "utils.h" // 引用我们的工具函数

int main() {
    // // 1. 准备输入数据
    // Tensor A;
    // A.shape = {2, 3};
    // A.data = {1, 2, 3, 4, 5, 6};

    // Tensor B;
    // B.shape = {3, 2};
    // B.data = {7, 8, 9, 10, 11, 12};

    // // 2. 调用 matmul 函数
    // std::cout << "Calculating C = A * B..." << std::endl;
    // Tensor C = matmul(A, B);
    // // std::cout<<A.shape[1];
    // // 3. 打印结果
    // std::cout << "\nResult Tensor C:" << std::endl;
    // print_tensor(C);

    // // 4. 手动验证 (我们的“单元测试”)
    // std::cout << "\nExpected Tensor C:" << std::endl;
    // std::cout << "58\t64" << std::endl;
    // std::cout << "139\t154" << std::endl;


// in main.cpp
Tensor Q;
Q.shape = {2, 2};
Q.data = {1, 2, 3, 4};

Tensor K;
K.shape = {2, 2};
K.data = {5, 6, 7, 8};

Tensor V; // V暂时用不到，但先创建好
V.shape = {2, 2};
V.data = {1, 0, 0, 1}; // V是一个单位矩阵


std::cout << "Calculating Self-Attention..." << std::endl;
Tensor result = self_attention(Q, K, V);
std::cout << "Ending Self-Attention..." << std::endl;

std::cout << "......last result......." << std::endl;
print_tensor(result);






















    return 0;
}