# A Handbook of CUDA Parallel Patterns

*Authored by jknb53, based on hands-on experience from Project Griffin.*

This document summarizes the four fundamental parallel patterns I implemented from scratch in C++/CUDA while building the core components of a Transformer.

## #1 Element-wise parallelism

* **summary:** Directly flatten the 2D matrix into a 1D array.
* **example:** gelu_kernel
* **key code:**

```cpp
# input:float* data  ,int rows ,int cols
int tid = threadIdx.x;
int global_tid = threadIdx.x + blockDim.x*blockIdx.x;
if(global_tid <rows*cols ){
	data[global_tid] = gelu(data[global_tid]) ;
}
# So wasteful, burns through so many threads. Feels like a brute-force miracle, the kind where you win with sheer numbers and don't care about the cannon fodder. Hmph.
```

---

## #2 Block-wise parallelism

* **summary:** Not enough hands, so the skilled do more.
* **example:** add_bias_kernel
* **key code:**

```cpp

  #input:float* data ,float* bias ,int rows int cols
  int tid = threadIdx.x;
  int tid_start = blockIdx.x*cols;
  //if(tid < cols){}
  for(int i = tid;i<cols;i+=blockDim.x){
  	data[tid_start + i] += bias[i];
  }
  // A marvelous feeling. The logic feels deeply tied to the size of `blockDim`, which is a variable parameter.
```

---

## #3 2D-matrix parallelism

* **summary:** picture followed(to be done)
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
  // The K parameter isn't fixed, which means a single thread could have a massive workload.
```

---

## #4 Parallel reduction

* **summary:** Use shared memory to calculate intermediate results; the rest is not much different from a serial implementation.
* **example:** layernorm_kernel
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
  __syncthread();//when they all eat over. My rule for `__syncthreads()`: first, you have to be careful when using shared memory. And second, I feel it's for when you need to reuse a result in shared memory from a previous step.
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
