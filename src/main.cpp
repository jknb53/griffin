#include "utils.h"
#include <iostream>

int main() {
    try {
        // 1. 准备比赛数据
        Tensor input(2, 4);
        input.data = {1, 2, 3, 3, 1, 5, 2, 8};
        //w1,w2
        Tensor w1(4, 8);
        w1.data = {1, 2, 3, 3, 1, 5, 2, 8,
                            1 ,2 ,3 ,4 ,5 ,6 ,7 ,8,
                            2, 4 ,89,9 ,2 ,7 , 0 ,7,
                            2, 4 ,5 ,9 ,12 ,7 , 2 ,7};
        Tensor w2(8, 4);
        w2.data = {1, 2, 3, 3, 
                            1, 5, 2, 8,
                            2, 4 ,5 ,9 ,
                            12 ,7 , 2 ,7,
                            1, 2, 3, 3, 
                            1, 5, 2, 8,
                            8 ,2 ,5 ,8 ,
                            6 ,1 ,22, 0};
        //b1,b2
        Tensor b1(1,8);
        b1.data= {1, 2, 3, 3, 1, 5, 2, 8};
        Tensor b2(1, 4);
        b2.data = {1, 2, 3, 3};

        // 2. CPU选手上场
        Tensor C_cpu = ffn_cpu(input,w1,b1,w2, b2);

        // // 3. GPU选手上场
        Tensor C_cuda = ffn_cuda(input,w1,b1,w2, b2);

        // 4. 打印双方结果
        std::cout << "--- CPU result ---" << std::endl;
        print_tensor(C_cpu);
        std::cout << "\n--- CUDA result ---" << std::endl;
        print_tensor(C_cuda);

        // 5. 裁判宣布比赛结果
        if (compare_tensors(C_cpu, C_cuda)) {
            std::cout << "\n[SUCCESS] The calculation results of the CPU and CUDA are not consistent!" << std::endl;
        } else {
            std::cout << "\n[FAILURE] The calculation results of the CPU and CUDA are not consistent!" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "An error occurred in the program.: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
