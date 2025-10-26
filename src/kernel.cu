#include<cuda_runtime.h>
#include "utils.h"


 __global__ void add_one_kernel(float* d_data, int n){

    size_t idx = blockIdx.x * blockDim.x +threadIdx.x;

    if(idx < n){
        d_data[idx] += 1.0f;
    }

}

// 定义一个普通的 C++ 函数作为桥梁
void launch_add_one_kernel(float* d_data, int n) {
    // 在 .cu 文件里，nvcc 编译器认识这个语法
    add_one_kernel<<<1, n>>>(d_data, n);
    CUDA_CHECK(cudaGetLastError());
}

//---------------------------------------------------------------------------------------
 __global__ void matmul_kernel(const float* d_A, const float* d_B, float* d_C, int M, int K, int N){
    //thread also idex
    size_t rows = blockIdx.y * blockDim.y +threadIdx.y;
    size_t cols = blockIdx.x * blockDim.x +threadIdx.x;

    if(rows< M&&cols<N){
        float sum = 0.0f;
        for(int i = 0;i<K;++i){
            sum += d_A[rows*K +i]*d_B[i*N +cols];
        }
        d_C[rows * N+cols] = sum;
    }
}




Tensor matmul_cuda(const Tensor& A, const Tensor& B) {
    // 【做】的部分：你的所有核心逻辑将在这里展开
    //init
    size_t M =A.rows;
    size_t K = A.cols;
    size_t N = B.cols;

    Tensor C(M,N);
    

    size_t size_a = A.data.size()*sizeof(float);
    size_t size_b = B.data.size()*sizeof(float);
    size_t size_c = C.data.size()*sizeof(float);

    float* d_a;
    float* d_b;
    float* d_c;

    //malloc
    CUDA_CHECK(cudaMalloc(&d_a,size_a));
    CUDA_CHECK(cudaMalloc(&d_b,size_b));
    CUDA_CHECK(cudaMalloc(&d_c,size_c));
    //copy
    CUDA_CHECK(cudaMemcpy(d_a,A.data.data(),size_a,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,B.data.data(),size_b,cudaMemcpyHostToDevice));
    //config
    dim3 threadPerBlock(16,16);
    dim3 numBlock((N+threadPerBlock.x-1)/threadPerBlock.x,(M+threadPerBlock.y-1)/threadPerBlock.y);

    matmul_kernel<<<numBlock,threadPerBlock>>>(d_a,d_b,d_c,M,K,N);
    CUDA_CHECK(cudaGetLastError());
    //copy back
    CUDA_CHECK(cudaMemcpy(C.data.data(),d_c,size_c,cudaMemcpyDeviceToHost));
    //free 
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return C;
}
//---------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------
__global__ void scale_kernel(float* data, float scale_factor , int n){
    //1.idx
    int idx = blockIdx.x * blockDim.x +threadIdx.x;
    if(idx < n){
        data[idx] /= scale_factor;
    }

}

extern "C" __global__ void softmax_kernel(float* data, int rows ,int cols){
   int row_idx = blockIdx.x;
   int row_start = row_idx * cols;
    if(row_idx < rows){
        float max_val = -INFINITY;
        if(threadIdx.x == 0){
            for(int i = 0;i<cols;++i){
                if(max_val <= data[row_start+i]){
                    max_val = data[row_start+i];
                }
            }
            for(int i = 0;i<cols;++i){
                data[row_start+i] -= max_val;
            }
            float sum = 0.0f;
            for(int i = 0;i<cols;++i){
                sum += expf(data[row_start+i]);
            }
            for(int i = 0;i<cols;++i){
                data[row_start+i] = expf(data[row_start+i])/sum;
            }
            //optimise，repeat expf many times，undone...？？？remember
        }
    }
}



Tensor self_attention_cuda_v2(const Tensor& Q, const Tensor& K,const Tensor& V){
    //---input validation ---
    if(Q.cols!=K.rows){
        throw std::invalid_argument("Dimension dismatch inself-attention :Q.cols must be equal to K.rows");
    }
        if(K.cols!=V.rows){
        throw std::invalid_argument("Dimension dismatch inself-attention :K.cols must be equal to V.rows");
    }


    size_t M = Q.rows;
    size_t dim_c1 = Q.cols;
    size_t dim_c2 = K.cols;
    size_t N = V.cols;

    Tensor S(M,dim_c2);
    Tensor A(M,N);


    float* d_q;
    float* d_k;
    float* d_v;  
    float* d_s;
    float* d_a;

    size_t size_q = Q.data.size()*sizeof(float);
    size_t size_k = K.data.size()*sizeof(float);
    size_t size_v = V.data.size()*sizeof(float);
    size_t size_s = S.data.size()*sizeof(float);
    size_t size_a= A.data.size()*sizeof(float);

    //malloc
    CUDA_CHECK(cudaMalloc(&d_q,size_q));
    CUDA_CHECK(cudaMalloc(&d_k,size_k));
    CUDA_CHECK(cudaMalloc(&d_v,size_v));
    CUDA_CHECK(cudaMalloc(&d_s,size_s));
    CUDA_CHECK(cudaMalloc(&d_a,size_a));

    //copy
    CUDA_CHECK(cudaMemcpy(d_q,Q.data.data(),size_q,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k,K.data.data(),size_k,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v,V.data.data(),size_v,cudaMemcpyHostToDevice));

    //config
    dim3 threadPerBlock(16,16);
    dim3 numBlock((N+threadPerBlock.x-1)/threadPerBlock.x,(M+threadPerBlock.y-1)/threadPerBlock.y);

    matmul_kernel<<<numBlock,threadPerBlock>>>(d_q,d_k,d_s,M,dim_c1,dim_c2);
    CUDA_CHECK(cudaGetLastError());   

    float scale_factor = sqrt(K.cols);
    int threadPerBlock_1D =  256;
    int block_num_1D = (M*dim_c2+threadPerBlock_1D-1)/threadPerBlock_1D;
    scale_kernel<<<block_num_1D,threadPerBlock_1D>>>(d_s,scale_factor,M*dim_c2);
    int threadPerBlock_soft = 32;
    int block_num_soft = S.rows ;
    softmax_kernel<<<block_num_soft, threadPerBlock_soft>>>(d_s,S.rows,S.cols);
    matmul_kernel<<<numBlock,threadPerBlock>>>(d_s,d_v,d_a,M,dim_c2,N);
    CUDA_CHECK(cudaGetLastError());   

 //copy back
CUDA_CHECK(cudaMemcpy(A.data.data(),d_a,size_a,cudaMemcpyDeviceToHost));

//malloc
CUDA_CHECK(cudaFree(d_q));
CUDA_CHECK(cudaFree(d_k));
CUDA_CHECK(cudaFree(d_v));
CUDA_CHECK(cudaFree(d_s));
CUDA_CHECK(cudaFree(d_a));
return A;
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
