#include "utils.h"
#include <cmath>
#include <limits>

void print_tensor(const Tensor& t) {
    // 假设是二维张量来打印
    size_t rows = t.rows;
    size_t cols = t.cols;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << t.data[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
}


Tensor matmul_cpu(const Tensor& A, const Tensor& B){
    //basic def
    size_t a_r = A.rows;
    size_t a_c = A.cols;

    size_t b_c = B.cols;

    Tensor C(a_r,b_c);

    //matmul
    if(a_c!=B.rows){
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    
    for(int i = 0; i<a_r;++i){
        for(int j = 0; j<b_c;++j){
            for(int k = 0; k<a_c;++k){
                    C.data[i*b_c + j] += A.data[i*a_c + k] *B.data[k*b_c + j];
            }
        }
    }
    return C;
}

Tensor softmax(const Tensor&input){
    Tensor result = input;// do not change input
    for(int i=0;i<result.rows;++i){
        //1.max
        float max = - std::numeric_limits<float>::infinity();
        for(int j =0; j <result.cols;++j){
            if(max <result.data[i*result.cols +j]){
                max =result.data[i*result.cols +j];
            }
        }
        // std::cout<<max<<std::endl;
        //2.sub&exp()
        for(int j=0;j<result.cols;++j){
            result.data[i*result.cols +j] -=max;
            result.data[i*result.cols +j] = std::exp(result.data[i*result.cols +j]);  
        }
        //3.sum cal
        float sum = 0;
        for(int j=0;j<result.cols;++j){
            sum +=result.data[i*result.cols +j];
        }

        //4.div
        for(int j=0;j<result.cols;++j) {
            result.data[i*result.cols +j] /= sum;
        }
    }
    return result;
}


Tensor self_attention(const Tensor& Q, const Tensor& K, const Tensor& V){
    Tensor scores =matmul_cpu(Q,K);
    float scale_factor = sqrt(K.cols);
    // for(int i = 0;i < K.rows*K.cols;++i)
    for(float&value :scores.data)
    {
        value /= scale_factor;
    }
    // ... 计算 scores 的代码 ...

    Tensor attention_weights = softmax(scores);

    // --- 验证步骤 ---
    // std::cout << "--- Intermediate: Attention Weights (Softmax output) ---" << std::endl;
    // print_tensor(attention_weights);

    // std::cout<<scale_factor<<std::endl;
    // print_tensor(scores);
    Tensor output_weights = matmul_cpu(attention_weights,V);
    return  output_weights;
}

bool compare_tensors(const Tensor& a, const Tensor& b, float tolerance ) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (size_t i = 0; i < a.data.size(); ++i) {
        if (std::abs(a.data[i] - b.data[i]) > tolerance) {
            std::cerr << "元素不匹配，位置 " << i << ": " << a.data[i] << " (CPU) vs " << b.data[i] << " (GPU)" << std::endl;
            return false;
        }
    }
    return true;
}

