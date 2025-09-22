// --- 【抄写块 3】---
#include "utils.h"
#include <iostream>

int main() {
    try {
        // 1. 准备比赛数据
        Tensor A(2, 2);
        A.data = {1, 2, 3, 4};
        Tensor B(2, 2);
        B.data = {5, 6, 7, 8};

        // 2. CPU选手上场
        Tensor C_cpu = matmul_cpu(A, B);

        // 3. GPU选手上场
        Tensor C_cuda = matmul_cuda(A, B);

        // 4. 打印双方结果
        std::cout << "--- CPU 运行结果 ---" << std::endl;
        print_tensor(C_cpu);
        std::cout << "\n--- CUDA 运行结果 ---" << std::endl;
        print_tensor(C_cuda);

        // 5. 裁判宣布比赛结果
        if (compare_tensors(C_cpu, C_cuda)) {
            std::cout << "\n[SUCCESS] CPU 和 CUDA 计算结果一致！" << std::endl;
        } else {
            std::cout << "\n[FAILURE] CPU 和 CUDA 计算结果不匹配！" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "程序发生错误: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
