// --- 【抄写块 2】---
#include "utils.h"
#include <iostream>

int main() {
    try {
        Tensor Q(2, 2); Q.data = {1, 2, 3, 4};
        Tensor K(2, 2); K.data = {5, 6, 7, 8};
        Tensor V(2, 2); V.data = {1, 0, 0, 1};

        // 1. CPU选手作为黄金标准
        Tensor C_cpu = self_attention(Q, K, V);

        // 2. 终极的纯血GPU选手上场
        Tensor C_cuda_v2 = self_attention_cuda_v2(Q, K, V);

        // 3. 打印双方结果
        std::cout << "--- CPU Self-Attention 运行结果 ---" << std::endl;
        print_tensor(C_cpu);
        std::cout << "\n--- CUDA Self-Attention v2 (纯血版) 运行结果 ---" << std::endl;
        print_tensor(C_cuda_v2);

        // 4. 裁判宣布比赛结果
        if (compare_tensors(C_cpu, C_cuda_v2)) {
            std::cout << "\n[SUCCESS] CPU 和 CUDA Self-Attention v2 计算结果一致！" << std::endl;
        } else {
            std::cout << "\n[FAILURE] CPU 和 CUDA Self-Attention v2 计算结果不匹配！" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "程序发生错误: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}