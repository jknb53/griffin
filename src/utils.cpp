#include "utils.h" // 包含我们自己的头文件
#include <cmath>
#include <limits>
#include <cuda_runtime.h>
#include <stdexcept>


// 在.cpp文件中，我们需要再次声明核函数，以便C++编译器知道它的存在
// extern "C" 是为了防止C++的命名修饰(name mangling)
// extern "C" __global__ void add_kernel(const float* d_a, const float* d_b, float* d_c, int n);



// prsize_t_tensor 函数的具体实现
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

// matmul_cpu 函数的具体实现
// !!! 注意：这里是空的，这正是你需要完成的部分 !!!
// Tensor matmul_cpu(const Tensor& A, const Tensor& B) {
//    //basic def
//     size_t a_height = A.rows;
//     size_t a_width = A.cols;
//     std::vector<float> a_mat = A.data;

//     size_t b_height =  B.rows;
//     size_t b_width = B.cols;
//     std::vector<float> b_mat = B.data;

//     Tensor C (a_height,b_width);
//     // C.shape = {a_height,b_width};   wrong,modified
//     //attention!!! we did not creat before
//     // C.rows = a_height;
//     // C.cols = b_width;
//     C.data.resize(C.rows*C.cols,0.0f);


// for(size_t i = 0;i < a_height ; ++i){
//     for(size_t j = 0;j < b_width ; ++j){
//         for(size_t k = 0;k < a_width ; ++k ){
//             C.data[i*C.cols + j] += a_mat[i*a_width + k] * b_mat[k*b_width + j];
//         }
//     }
// }
//     // 这是一个临时的、错误的实现，以便项目能编译通过
//     // 你需要把它替换成你真正的代码
//     // std::cout << "[ERROR] matmul_cpu function is not implemented yet!" << std::endl;
//     return C; // 返回一个空的Tensor
// }


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

Tensor self_attention_cuda(const Tensor& Q, const Tensor& K,const Tensor& V){
    Tensor scores  = matmul_cuda(Q,K);
    float scale_factor = sqrt(K.cols);
    for(float& value : scores.data){
        value/= scale_factor;
    }
    Tensor attention_weights = softmax(scores);
    Tensor output = matmul_cuda(attention_weights, V);
    return output;
}


