#include "utils.h" // 包含我们自己的头文件
#include <cmath>
#include <limits>

// prsize_t_tensor 函数的具体实现
void print_tensor(const Tensor& t) {
    // 假设是二维张量来打印
    size_t rows = t.shape[0];
    size_t cols = t.shape[1];
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << t.data[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
}

// matmul 函数的具体实现
// !!! 注意：这里是空的，这正是你需要完成的部分 !!!
Tensor matmul(const Tensor& A, const Tensor& B) {
   //basic def
    size_t a_height = A.shape[0];
    size_t a_width = A.shape[1];
    std::vector<float> a_mat = A.data;

    size_t b_height =  B.shape[0];
    size_t b_width = B.shape[1];
    std::vector<float> b_mat = B.data;

    Tensor C ;
    C.shape = {a_height,b_width};
    //attention!!! we did not creat before
    // C.shape[0] = a_height;
    // C.shape[1] = b_width;
    C.data.resize(C.shape[0]*C.shape[1],0.0f);


for(size_t i = 0;i < a_height ; ++i){
    for(size_t j = 0;j < b_width ; ++j){
        for(size_t k = 0;k < a_width ; ++k ){
            C.data[i*C.shape[1] + j] += a_mat[i*a_width + k] * b_mat[k*b_width + j];
        }
    }
}
    // 这是一个临时的、错误的实现，以便项目能编译通过
    // 你需要把它替换成你真正的代码
    // std::cout << "[ERROR] matmul function is not implemented yet!" << std::endl;
    return C; // 返回一个空的Tensor
}


Tensor softmax(const Tensor&input){
    Tensor result = input;// do not change input
    for(int i=0;i<result.shape[0];++i){
        //1.max
        float max = - std::numeric_limits<float>::infinity();
        for(int j =0; j <result.shape[1];++j){
            if(max <result.data[i*result.shape[1] +j]){
                max =result.data[i*result.shape[1] +j];
            }
        }
        // std::cout<<max<<std::endl;
        //2.sub&exp()
        for(int j=0;j<result.shape[1];++j){
            result.data[i*result.shape[1] +j] -=max;
            result.data[i*result.shape[1] +j] = std::exp(result.data[i*result.shape[1] +j]);  
        }
        //3.sum cal
        float sum = 0;
        for(int j=0;j<result.shape[1];++j){
            sum +=result.data[i*result.shape[1] +j];
        }

        //4.div
        for(int j=0;j<result.shape[1];++j) {
            result.data[i*result.shape[1] +j] /= sum;
        }
    }
    return result;
}
Tensor self_attention(const Tensor& Q, const Tensor& K, const Tensor& V){
    Tensor scores =matmul(Q,K);
    float scale_factor = sqrt(K.shape[1]);
    // for(int i = 0;i < K.shape[0]*K.shape[1];++i)
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
    Tensor output_weights = matmul(attention_weights,V);
    return  output_weights;
}