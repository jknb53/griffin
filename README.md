```markdown
# 🚀 Project Griffin | Griffin 项目

*Bilingual README: Jump to [English](#-english) | [中文](#-中文)*

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
    -   [x] `layernorm_cpu`: Layer normalization with configurable gamma and beta parameters.
    -   [x] `gelu_cpu`: Gaussian Error Linear Unit activation function.
    -   [x] `add_bias_cpu`: Bias addition operation with broadcasting support.
    -   [x] `ffn_cpu`: Complete FeedForward Network implementation.

-   **GPU Operators (`kernel.cu`):**
    -   [x] `matmul_cuda`: A pure-GPU matrix multiplication implementation.
    -   [x] `self_attention_cuda_v2`: A high-performance, pure-GPU version of self-attention that orchestrates all computations on the device to eliminate CPU-GPU data roundtrips.
        -   Includes custom kernels: `scale_kernel` and a simplified `softmax_kernel`.
    -   [x] `layernorm_cuda`: GPU-accelerated layer normalization with parallel reduction optimization.
    -   [x] `gelu_cuda`: GPU implementation of GELU activation with custom CUDA kernel.
    -   [x] `add_bias_cuda`: GPU bias addition kernel with efficient memory access patterns.
    -   [x] `ffn_cuda`: Complete GPU pipeline for FeedForward Network with optimized memory management.

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

The main executable runs a comprehensive test to verify the correctness of the FeedForward Network implementation, which includes all core operators working in sequence.

```bash
./griffin
```

A `[SUCCESS]` message indicates that the CPU and GPU implementations of the complete FFN pipeline produce matching results, validating the correctness of:
- Matrix multiplication (`matmul`)
- Bias addition (`add_bias`) 
- GELU activation function (`gelu`)
- The complete FeedForward Network orchestration (`ffn`)

#### 🏗️ Architecture Highlights

**Rigorous Development Philosophy:**
- **"CPU Defines Truth"**: Every GPU implementation is validated against its CPU counterpart using strict numerical comparison.
- **Incremental Complexity**: Starting with individual operators and building up to complete neural network components.
- **Memory Management Mastery**: Explicit CUDA memory management demonstrates deep understanding of GPU computing principles.

**Key Technical Achievements:**
- **Pure GPU Pipelines**: The `ffn_cuda` implementation orchestrates an entire computation graph on GPU without CPU-GPU round trips.
- **Parallel Reduction Optimization**: LayerNorm uses shared memory and parallel reduction for efficient variance computation.
- **Modular Design**: Each operator can be tested, verified, and reused independently.

#### 🗺️ Future Roadmap (Next Steps)

- [x] ~~Implement `LayerNorm` Kernel.~~
- [ ] Optimize `softmax_kernel` with parallel reduction.
- [ ] Optimize `matmul_kernel` with Shared Memory and tiled matrix multiplication.
- [x] ~~Implement GELU Activation Kernel.~~
- [ ] Assemble a full GPT-2 Transformer Block (combining Self-Attention, LayerNorm, and FFN).
- [ ] Implement Multi-Head Attention mechanism.
- [ ] Build positional encoding and input embeddings.
- [ ] Create the final GPT-2 Model for inference.
- [ ] Add model loading capabilities (weights from pre-trained checkpoints).
- [ ] Implement text tokenization and generation pipeline.

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
  - [X] `layernorm_cpu`: 带有可配置gamma和beta参数的层归一化。
  - [X] `gelu_cpu`: 高斯误差线性单元激活函数。
  - [X] `add_bias_cpu`: 支持广播机制的偏置加法操作。
  - [X] `ffn_cpu`: 完整的前馈神经网络实现。
- **GPU 算子 (`kernel.cu`):**

  - [X] `matmul_cuda`: 纯GPU实现的矩阵乘法。
  - [X] `self_attention_cuda_v2`: 一个高性能的、纯GPU版本的自注意力实现，通过将所有计算保留在设备端，消除了不必要的CPU-GPU数据往返。
    - 包含自定义核函数：`scale_kernel` 和一个简化的 `softmax_kernel`。
  - [X] `layernorm_cuda`: 带有并行规约优化的GPU加速层归一化。
  - [X] `gelu_cuda`: 带有自定义CUDA核函数的GELU激活函数GPU实现。
  - [X] `add_bias_cuda`: 具有高效内存访问模式的GPU偏置加法核函数。
  - [X] `ffn_cuda`: 具有优化内存管理的完整前馈神经网络GPU流水线。
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

生成的可执行文件将运行一个综合测试，验证前馈神经网络实现的正确性，其中包含所有核心算子按顺序工作的情况。

```bash
./griffin
```

输出 `[SUCCESS]` 信息代表CPU和GPU的完整FFN流水线实现结果一致，验证了以下组件的正确性：
- 矩阵乘法 (`matmul`)
- 偏置加法 (`add_bias`)
- GELU激活函数 (`gelu`)
- 完整前馈神经网络编排 (`ffn`)

#### 🏗️ 架构亮点

**严谨的开发理念：**
- **"CPU定义真理"**：每个GPU实现都通过与其CPU对应版本进行严格的数值比较来验证。
- **增量式复杂度**：从单个算子开始，逐步构建完整的神经网络组件。
- **内存管理精通**：显式的CUDA内存管理展示了对GPU计算原理的深度理解。

**关键技术成就：**
- **纯GPU流水线**：`ffn_cuda` 实现在GPU上编排整个计算图，无需CPU-GPU往返传输。
- **并行规约优化**：LayerNorm使用共享内存和并行规约进行高效的方差计算。
- **模块化设计**：每个算子都可以独立测试、验证和重用。

#### 🗺️ 未来路线图 (下一步计划)

- [X] ~~实现 `LayerNorm` 的CUDA核函数。~~
- [ ] 使用并行规约 (Parallel Reduction) 算法优化 `softmax_kernel`。
- [ ] 使用共享内存 (Shared Memory) 和分块矩阵乘法优化 `matmul_kernel`。
- [X] ~~实现 `GELU` 激活函数的CUDA核函数。~~
- [ ] 将所有算子组装成一个完整的GPT-2 Transformer模块 (Block)。
- [ ] 实现多头注意力机制 (Multi-Head Attention)。
- [ ] 构建位置编码和输入嵌入层。
- [ ] 最终构建出可用于推理的完整GPT-2模型。
- [ ] 添加模型加载功能 (从预训练检查点加载权重)。
- [ ] 实现文本标记化和生成流水线。
