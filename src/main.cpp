// --- 【抄写块 3】---
#include "utils.h"
#include <iostream>

int main() {
    try {
        // 1. 准备比赛数据
        Tensor input(2, 4);
        input.data = {1, 2, 3, 3, 1, 5, 2, 8};

        Tensor gamma(1,4);
        gamma.data = {1,2,3,4};

        Tensor beta(1,4);
        beta.data = {1,2,3,4};

        // 2. CPU选手上场
        Tensor C_cpu = layernorm_cpu(input,gamma,beta);

        // // 3. GPU选手上场
        Tensor C_cuda = layernorm_cuda(input,gamma,beta);

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
