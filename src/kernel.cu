#include<cuda_runtime.h>
#include "utils.h"


 extern "C" __global__ void add_one_kernel(float* d_data, int n){

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


extern "C" __global__ void matmul_kernel(const float* d_A, const float* d_B, float* d_C, int M, int K, int N){
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