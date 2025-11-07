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

Tensor layernorm_cpu(const Tensor &input , const Tensor &gamma, const Tensor &beta){
    Tensor output = input;
    std::vector<float> mean(output.rows) ,var(output.rows);
    float esp = 1e-5f;
    //1.mean 
    for(int i =0; i<output.rows;++i){
        float sum =0;//attention must float
        for(int j =0; j<output.cols;++j){
                    sum+=output.data[i*output.cols+j];
        }
        mean[i] = sum/output.cols;
    }
    //2.var**2
    for(int i =0; i<output.rows;++i){
        float sum_diff =0;
  
        for(int j =0; j<output.cols;++j){
                    sum_diff+=pow((output.data[i*output.cols+j]-mean[i]),2);
        }
        var[i] =sum_diff/output.cols;
    
    }  
    //3.norm and multiply gamma and beta
    for(int i =0; i<output.rows;++i){
        for(int j =0; j<output.cols;++j){
            output.data[i*output.cols+j]= gamma.data[j]*(output.data[i*output.cols+j]-mean[i])/sqrt(var[i]+esp)+beta.data[j];
        }
    }
    return output;
}

    Tensor gelu_cpu(const Tensor & input){
        Tensor copy = input;

        size_t rows =copy.rows;
        size_t cols =copy.cols;
        for(int i = 0;i<rows*cols;++i){
            copy.data[i]=0.5*copy.data[i]*(1+std::tanh(sqrt(2/M_PI)*(copy.data[i]+copy.data[i]*copy.data[i]*copy.data[i]*0.044715)));
        }
        return copy;
}

//--------------------------------ffn---------------------------
Tensor add_bias_cpu(const Tensor & hidden,const Tensor & bias){
    Tensor output =hidden;//every time i copy a copy??? 
    //other situation ,to be done......
    // if(hidden.cols!=bias.cols){
        
    // }
    //right side dimension is same
    if(hidden.cols==bias.cols){
        for(int i =0;i<hidden.rows;++i){
            for(int j =0;j<hidden.cols;++j){
                output.data[i*hidden.cols+j] += bias.data[j];
            }
        }
    }
    return output;
}


Tensor ffn_cpu(const Tensor & input,const Tensor & w1,const Tensor & b1,const Tensor & w2,const Tensor  & b2){
    Tensor hidden1 = matmul_cpu(input,w1);
    Tensor hidden2 = add_bias_cpu(hidden1,b1);
    Tensor hidden3 = gelu_cpu(hidden2);
    Tensor output1 = matmul_cpu(hidden3,w2);
    Tensor output2 = add_bias_cpu(output1,b2);

    return output2;
}