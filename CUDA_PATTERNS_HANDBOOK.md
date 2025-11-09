# MY_CUDA_PARALLEL_SUMMARY

## #1逐元素并行(element-wise parallelism)

* **summary：直接把它二维矩阵直接拉直，拉成一维数组**
* **example：gelu_kernel**
* **key code:**

```cpp
# input:float* data  ,int rows ,int cols
int tid = threadIdx.x;
int global_tid = threadIdx.x + blockDim.x*blockIdx.x;
if(global_tid <rows*cols ){
	data[global_tid] = gelu(data[global_tid]) ;
}
#太浪费了，浪费了很多线程，感觉就是大力出奇迹，只要人多不怕炮灰死的感觉,哼哼
```

---

## #2分块跨网格并行(block-wise parallelism)

* **summary:人手不够，能者多劳**
* **example:add_bias_kernel**
* **key code:**
  ```cpp

  #input:float* data ,float* bias ,int rows int cols
  int tid = threadIdx.x;
  int tid_start = blockIdx.x*cols;
  //if(tid < cols){}
  for(int i = tid;i<cols;i+=blockDim.x){
  	data[tid_start + i] += bias[i];
  }
  //很奇妙感觉，感觉及时跟blockDim大小息息相关，而这是个可变参数
  ```

---

## #3二维矩阵并行(2D-matrix parallelism)

* **summary：picture followed(to be done)**
* **example:** matmul_kernel
* **key_code:**
  ```cpp
  #input: float* mat1,float* mat2,float* ouput,int M,int K ,int N
  int tidx = blockIdx.x*blockDim.x + threadIdx.x ;
  int tidy = blockIdx.y*blockDim.y + threadIdx.y ;

  if(tidx < N&& tidy < M){
  	float sum = 0.0f;
  	for(int coDim = 0;coDim < K; coDim++){
  	sum += mat1[tidy * K + coDim]*mat2[coDim * N + tidx];
  }
  output[tidy * N + tidx] = sum;
  }
  //K参数未定，一个线程可能做的工作量很大
  ```

---

## #4并行规约(parallel reduction)

* **summary：用共享内存单独计算中间结果,其他与串行差别不大**
* **example：layernorm_kernel**
* **key_code:**
  ```cpp
  #input:float* data,float* gamma , float* beta ,int rows ,int cols 
  int tid = threadIdx.x;
  int tid_start = blockIdx.x*cols;
  int global_tid = blockIdx.x*blockDim.x + threadIdx.x;
  float eps = 1e-5;
  float sum_mean = 0.0f;//pocket1
  float sum_var = 0.0f;//pocket2

  __shared__ float s_a[256];//military camp
  //preprocess
  for(int i = tid;i < cols;i += blockDim.x){
  	sum_mean += data[tid_start + i];
  }
  s_a[tid] = sum_mean;

  //mean_reduction
  for(int stride = blockDim.x>>1;stride != 0;stride /=2){
  	if(tid < stride){
  		s_a[tid] += s_a[tid + stride];
  }
  __syncthread();//when they all eat over 线程中我觉得用到__syncthread()是我觉得首先就是共享内存用到时候注意，然后的话我感觉就是你要多次复用共享内存的前面结果
  }
  //scheme 1
  if(tid == 0){
  	s_a[0] /= cols;
  }
  __syncthreads();
  float mean = s_a[0];
  //scheme 2
  //float mean = s_a[0]/cols;
  for(int i = tid ;i <cols ;i+=blockDim.x){
  	sum_var  += (data[tid_start + i]-mean)*(data[tid_start + i]-mean);
  }
  s_a[tid] = sum_var;

  //var_reduction
  for(int stride = blockDim.x>>1;stride != 0;stride /=2){
  	if(tid < stride){
  		s_a[tid] += s_a[tid + stride];
  }
  __syncthread();
  }
  if(tid == 0){
  	s_a[0] /=cols;
  }
  __syncthread();
  float var = s_a[0];
  float div = 1/sqrt(var+eps);
  //transform
  for(int i = tid ;i <cols ;i+=blockDim.x){
  	data[tid_start +i]  = gamma[i]*(data[tid_start + i]-mean)*div+beta[i];
  }
  ```
