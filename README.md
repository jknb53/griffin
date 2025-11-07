```markdown
# 🚀 Project Griffin | Griffin 项目

*Read this in other languages: [English](./README.md) | [中文](./README.zh-CN.md)* 
*(Note: For a bilingual README in a single file, you can remove the line above and the corresponding Chinese README file.)*

---

### 🇬🇧 English

A lightweight deep learning inference framework built from scratch in C++ and CUDA. This project is a hands-on implementation following the principles of Andrej Karpathy's "Neural Networks: Zero to Hero" course, with a focus on understanding the first principles of modern deep learning models like GPT-2.

#### 🎯 Project Goals

The primary objective of this project is not to build a production-ready library, but to serve as a rigorous, practice-based learning journey. The key goals are:

-   **Deepen Understanding of Transformers:** Deconstruct high-level concepts like Self-Attention into their fundamental mathematical operators and implement them from scratch.
-   **Master C++/CUDA Programming:** Gain proficiency in C++/CUDA for high-performance computing, including memory management, kernel programming, and orchestrating complex computation flows on the GPU.
-   **Embrace Professional Engineering Practices:** Adhere to a strict workflow using modern CMake for building, Git for version control (following a feature-branching model), and a TDD-like approach by verifying every CUDA implementation against a "golden standard" CPU version.

#### ✨ Features Implemented

This project is being built incrementally. The following core components have been successfully implemented and verified:

-   **Core Data Structure:**
    -   [x] A simple `Tensor` struct in C++ for handling multi-dimensional data.

-   **CPU Operators (`cpu_ops.cpp`):**
    -   [x] `matmul_cpu`: Naive matrix multiplication.
    -   [x] `softmax_cpu`: Numerically stable softmax, applied row-wise.
    -   [x] `self_attention`: A complete, verifiable CPU implementation of the self-attention mechanism.

-   **GPU Operators (`kernel.cu`):**
    -   [x] `matmul_cuda`: A pure-GPU matrix multiplication implementation.
    -   [x] `self_attention_cuda_v2`: A high-performance, pure-GPU version of self-attention that orchestrates all computations on the device to eliminate CPU-GPU data roundtrips.
        -   Includes custom kernels: `scale_kernel` and a simplified `softmax_kernel`.

-   **Build & Test System:**
    -   [x] A robust build system configured with CMake to handle C++/CUDA mixed compilation.
    -   [x] A testing framework within `main.cpp` to compare CPU and GPU outputs for correctness.

#### 🛠️ How to Build and Run

##### Prerequisites

-   A C++ compiler (g++)
-   NVIDIA CUDA Toolkit (nvcc)
-   CMake (version 3.10+)

##### Build Steps

Clone the repository and run the following commands from the project root directory:

```bash
mkdir build
cd build
cmake ..
make
```

##### Run Tests

The main executable runs a series of tests to verify the correctness of the implemented operators.

```bash
./griffin
```

A `[SUCCESS]` message indicates that the CPU and GPU implementations produce matching results.

#### 🗺️ Future Roadmap (Next Steps)

- [ ] Implement `LayerNorm` Kernel.
- [ ] Optimize `softmax_kernel` with parallel reduction.
- [ ] Optimize `matmul_kernel` with Shared Memory.
- [ ] Implement GELU Activation Kernel.
- [ ] Assemble a full GPT-2 Transformer Block.
- [ ] Build the final GPT-2 Model for inference.

---

### 🇨🇳 中文

一个从零开始，使用C++和CUDA构建的轻量级深度学习推理框架。本项目是对 Andrej Karpathy 的 "Neural Networks: Zero to Hero" 课程理念的亲手实践，专注于从第一性原理理解如GPT-2等现代深度学习模型的底层运作机制。

#### 🎯 项目目标

本项目的核心并非构建一个生产级的代码库，而是一次严格的、基于实践的刻意练习之旅。主要目标包括：

- **深化对Transformer的理解：** 将自注意力 (Self-Attention) 等高级概念，拆解为其最基础的数学算子，并从零开始实现它们。
- **掌握C++/CUDA编程：** 熟练运用C++/CUDA进行高性能计算，包括内存管理、核函数编程以及在GPU上编排复杂的计算流。
- **拥抱专业工程实践：** 遵循严谨的工作流，使用现代CMake构建项目，通过Git进行版本控制（遵循功能分支模型），并采用类似TDD的方法，将每一个CUDA实现与“黄金标准”的CPU版本进行正确性验证。

#### ✨ 已实现功能

本项目采用增量式开发。以下核心组件已被成功实现并通过验证：

- **核心数据结构:**

  - [X] 用于处理多维数据的简易 `Tensor` C++ 结构体。
- **CPU 算子 (`cpu_ops.cpp`):**

  - [X] `matmul_cpu`: 朴素的CPU矩阵乘法实现。
  - [X] `softmax_cpu`: 逐行应用的、数值稳定的CPU Softmax。
  - [X] `self_attention`: 一个完整的、可作为基准的CPU自注意力机制实现。
- **GPU 算子 (`kernel.cu`):**

  - [X] `matmul_cuda`: 纯GPU实现的矩阵乘法。
  - [X] `self_attention_cuda_v2`: 一个高性能的、纯GPU版本的自注意力实现，通过将所有计算保留在设备端，消除了不必要的CPU-GPU数据往返。
    - 包含自定义核函数：`scale_kernel` 和一个简化的 `softmax_kernel`。
- **构建与测试系统:**

  - [X] 使用CMake配置的、能够处理C++/CUDA混合编译的健壮构建系统。
  - [X] 在 `main.cpp` 中搭建的、用于对比CPU和GPU输出以验证正确性的测试框架。

#### 🛠️ 如何构建与运行

##### 环境要求

- C++ 编译器 (g++)
- NVIDIA CUDA Toolkit (nvcc)
- CMake (3.10+版本)

##### 构建步骤

克隆本仓库，并在项目根目录下执行以下命令：

```bash
mkdir build
cd build
cmake ..
make
```

##### 运行测试

生成的可执行文件将运行一系列测试，以验证已实现算子的正确性。

```bash
./griffin
```

输出 `[SUCCESS]` 信息代表CPU和GPU的实现结果一致。

#### 🗺️ 未来路线图 (下一步计划)

- [X] 实现 `LayerNorm` 的CUDA核函数。
- [ ] 使用并行规约 (Parallel Reduction) 算法优化 `softmax_kernel`。
- [ ] 使用共享内存 (Shared Memory) 优化 `matmul_kernel`。
- [X] 实现 `GELU` 激活函数的CUDA核函数。
- [ ] 将所有算子组装成一个完整的GPT-2 Transformer模块 (Block)。
- [ ] 最终构建出可用于推理的完整GPT-2模型。
