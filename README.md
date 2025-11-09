```markdown
# 🚀 Project Griffin | Griffin 项目

*Bilingual README: Jump to [English](#-english) | [中文](#-中文)*

## 📋 Phase I Status: COMPLETED ✅

**Phase I "Full-Stack Operator Forging" has been successfully completed!** All core deep learning operators have been implemented from scratch with both CPU and CUDA versions, achieving full functional parity and correctness validation.

---

### 🇬🇧 English

A lightweight deep learning inference framework built from scratch in C++ and CUDA. This project is a hands-on implementation following the principles of Andrej Karpathy's "Neural Networks: Zero to Hero" course, with a focus on understanding the first principles of modern deep learning models like GPT-2.

**🎉 Phase I Achievement**: Complete implementation of fundamental deep learning operators with rigorous CPU-GPU validation, modular architecture, and comprehensive error analysis documentation.

#### 🎯 Project Goals

The primary objective of this project is not to build a production-ready library, but to serve as a rigorous, practice-based learning journey. The key goals are:

-   **Deepen Understanding of Transformers:** Deconstruct high-level concepts like Self-Attention into their fundamental mathematical operators and implement them from scratch.
-   **Master C++/CUDA Programming:** Gain proficiency in C++/CUDA for high-performance computing, including memory management, kernel programming, and orchestrating complex computation flows on the GPU.
-   **Embrace Professional Engineering Practices:** Adhere to a strict workflow using modern CMake for building, Git for version control (following a feature-branching model), and a TDD-like approach by verifying every CUDA implementation against a "golden standard" CPU version.

#### ✨ Phase I Implementation Status

**All core components have been successfully implemented, tested, and documented:** ✅

-   **Core Data Structure:**
    -   [x] A simple `Tensor` struct in C++ for handling multi-dimensional data.

-   **CPU Operators (`cpu_ops.cpp`):** ✅ **PHASE I COMPLETE**
    -   [x] `matmul_cpu`: Naive matrix multiplication serving as the "golden standard".
    -   [x] `softmax_cpu`: Numerically stable softmax, applied row-wise.
    -   [x] `self_attention_cpu`: Complete, verifiable CPU implementation of the self-attention mechanism.
    -   [x] `layernorm_cpu`: Layer normalization with configurable gamma and beta parameters.
    -   [x] `gelu_cpu`: Gaussian Error Linear Unit activation function.
    -   [x] `add_bias_cpu`: Bias addition operation with broadcasting support.
    -   [x] `ffn_cpu`: Complete FeedForward Network implementation.

-   **GPU Operators (`kernel.cu`):** ✅ **PHASE I COMPLETE**
    -   [x] `matmul_cuda`: Pure-GPU matrix multiplication with custom kernel implementation.
    -   [x] `self_attention_cuda_v2`: High-performance, pure-GPU self-attention pipeline eliminating CPU-GPU roundtrips.
        -   Includes optimized kernels: `scale_kernel`, `softmax_kernel` with parallel reduction.
    -   [x] `layernorm_cuda`: GPU-accelerated layer normalization with shared memory optimization.
    -   [x] `gelu_cuda`: GPU GELU activation with element-wise parallelization.
    -   [x] `add_bias_cuda`: GPU bias addition with efficient block-stride memory access patterns.
    -   [x] `ffn_cuda`: Complete GPU pipeline orchestrating multi-kernel computation flow.

-   **Project Architecture & Engineering:** ✅ **PHASE I COMPLETE**
    -   [x] Modular codebase structure with separation of concerns (`cpu_ops.cpp`, `kernel.cu`, `utils.h`).
    -   [x] Robust CMake build system handling C++/CUDA mixed compilation.
    -   [x] Comprehensive testing framework with CPU-GPU numerical validation.
    -   [x] Professional Git workflow with feature branching and incremental development.

-   **Documentation & Knowledge Synthesis:** ✅ **PHASE I COMPLETE**
    -   [x] Complete technical review documentation (`playbook/phase1/`).
    -   [x] Error analysis and debugging methodology documentation.
    -   [x] Performance optimization insights and parallel computing pattern analysis.

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

#### 🏗️ Phase I Architecture Highlights

**Rigorous Development Philosophy:**

- **"CPU Defines Truth"**: Every GPU implementation is validated against its CPU counterpart using strict numerical comparison.
- **Incremental Complexity**: Starting with individual operators and building up to complete neural network components.
- **Memory Management Mastery**: Explicit CUDA memory management demonstrates deep understanding of GPU computing principles.
- **Error-Driven Learning**: Comprehensive debugging sessions documented for future reference and learning.

**Key Technical Achievements:**

- **Pure GPU Pipelines**: The `ffn_cuda` implementation orchestrates an entire computation graph on GPU without CPU-GPU round trips.
- **Parallel Computing Mastery**: Implementation of multiple parallel patterns:
  - 2D parallel (matrix multiplication)
  - 1D reduction parallel (LayerNorm with shared memory)
  - 1D element-wise parallel (GELU activation)
- **System Integration Excellence**: Successfully orchestrated multi-kernel workflows with proper memory lifecycle management.
- **Modular Architecture**: Clean separation between CPU reference implementations and GPU optimizations.

**Engineering Lessons Learned:**

- **"Debugging First Law"**: The most critical errors often hide in the most unexpected places.
- **Memory Hierarchy Optimization**: Understanding register vs shared memory vs global memory access patterns.
- **Asynchronous Execution Awareness**: Proper synchronization between CPU and GPU execution contexts.

#### 🗺️ Development Roadmap

**✅ Phase I: "Full-Stack Operator Forging" - COMPLETED**
- [x] Core tensor data structure and CPU reference implementations
- [x] GPU kernel implementations for all fundamental operators  
- [x] Complete FeedForward Network with multi-kernel orchestration
- [x] Comprehensive testing and validation framework
- [x] Modular architecture and professional engineering practices
- [x] Technical documentation and knowledge synthesis

**🚧 Phase II: "High-Performance System Integration" - UPCOMING**
- [ ] Advanced GPU memory optimization techniques
- [ ] Kernel fusion and computation graph optimization  
- [ ] Multi-Head Attention mechanism implementation
- [ ] Complete GPT-2 Transformer Block assembly
- [ ] Performance profiling and bottleneck analysis

**🔮 Phase III: "Production-Ready Inference Engine" - PLANNED**
- [ ] Model loading from pre-trained checkpoints (GPT-2)
- [ ] Text tokenization and preprocessing pipeline
- [ ] Batched inference optimization
- [ ] End-to-end text generation capabilities
- [ ] Performance benchmarking against production frameworks

---

### 🇨🇳 中文

一个从零开始，使用C++和CUDA构建的轻量级深度学习推理框架。本项目是对 Andrej Karpathy 的 "Neural Networks: Zero to Hero" 课程理念的亲手实践，专注于从第一性原理理解如GPT-2等现代深度学习模型的底层运作机制。

**🎉 第一阶段成果**: 完成了所有基础深度学习算子的从零实现，建立了严格的CPU-GPU验证体系，实现了模块化架构，并完成了全面的错误分析与技术总结文档。

#### 🎯 项目目标

本项目的核心并非构建一个生产级的代码库，而是一次严格的、基于实践的刻意练习之旅。主要目标包括：

- **深化对Transformer的理解：** 将自注意力 (Self-Attention) 等高级概念，拆解为其最基础的数学算子，并从零开始实现它们。
- **掌握C++/CUDA编程：** 熟练运用C++/CUDA进行高性能计算，包括内存管理、核函数编程以及在GPU上编排复杂的计算流。
- **拥抱专业工程实践：** 遵循严谨的工作流，使用现代CMake构建项目，通过Git进行版本控制（遵循功能分支模型），并采用类似TDD的方法，将每一个CUDA实现与“黄金标准”的CPU版本进行正确性验证。

#### ✨ 第一阶段实现状态

**所有核心组件已成功实现、测试并文档化:** ✅

- **核心数据结构:**
  - [x] 用于处理多维数据的简易 `Tensor` C++ 结构体。

- **CPU 算子 (`cpu_ops.cpp`):** ✅ **第一阶段完成**
  - [x] `matmul_cpu`: 作为"黄金标准"的朴素矩阵乘法实现。
  - [x] `softmax_cpu`: 逐行应用的、数值稳定的CPU Softmax。
  - [x] `self_attention_cpu`: 完整的、可作为基准验证的CPU自注意力机制实现。
  - [x] `layernorm_cpu`: 带有可配置gamma和beta参数的层归一化。
  - [x] `gelu_cpu`: 高斯误差线性单元激活函数。
  - [x] `add_bias_cpu`: 支持广播机制的偏置加法操作。
  - [x] `ffn_cpu`: 完整的前馈神经网络实现。

- **GPU 算子 (`kernel.cu`):** ✅ **第一阶段完成**
  - [x] `matmul_cuda`: 纯GPU实现的矩阵乘法，包含自定义核函数实现。
  - [x] `self_attention_cuda_v2`: 高性能纯GPU自注意力流水线，消除CPU-GPU往返传输。
    - 包含优化的核函数：`scale_kernel`、带并行规约的 `softmax_kernel`。
  - [x] `layernorm_cuda`: 带共享内存优化的GPU加速层归一化。
  - [x] `gelu_cuda`: 逐元素并行化的GPU GELU激活函数。
  - [x] `add_bias_cuda`: 采用高效块步长内存访问模式的GPU偏置加法。
  - [x] `ffn_cuda`: 编排多核函数计算流的完整GPU流水线。

- **项目架构与工程实践:** ✅ **第一阶段完成**
  - [x] 关注点分离的模块化代码结构 (`cpu_ops.cpp`, `kernel.cu`, `utils.h`)。
  - [x] 处理C++/CUDA混合编译的强健CMake构建系统。
  - [x] 带CPU-GPU数值验证的全面测试框架。
  - [x] 采用功能分支和增量开发的专业Git工作流。

- **文档化与知识总结:** ✅ **第一阶段完成**
  - [x] 完整的技术复盘文档 (`playbook/phase1/`)。
  - [x] 错误分析与调试方法论文档。
  - [x] 性能优化洞察与并行计算模式分析。

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

#### 🏗️ 第一阶段架构亮点

**严谨的开发理念：**

- **"CPU定义真理"**：每个GPU实现都通过与其CPU对应版本进行严格的数值比较来验证。
- **增量式复杂度**：从单个算子开始，逐步构建完整的神经网络组件。
- **内存管理精通**：显式的CUDA内存管理展示了对GPU计算原理的深度理解。
- **错误驱动学习**：全面的调试过程文档化，为未来参考和学习提供宝贵资料。

**关键技术成就：**

- **纯GPU流水线**：`ffn_cuda` 实现在GPU上编排整个计算图，无需CPU-GPU往返传输。
- **并行计算精通**：实现了多种并行模式：
  - 二维并行 (矩阵乘法)
  - 一维归约并行 (LayerNorm带共享内存)
  - 一维逐元素并行 (GELU激活函数)
- **系统集成卓越性**：成功编排多核函数工作流，正确管理内存生命周期。
- **模块化架构**：CPU参考实现与GPU优化之间的清晰分离。

**工程经验总结：**

- **"调试第一定律"**：最关键的错误往往隐藏在最意想不到的地方。
- **内存层次优化**：理解寄存器 vs 共享内存 vs 全局内存的访问模式。
- **异步执行意识**：CPU与GPU执行上下文之间的正确同步。

#### 🗺️ 开发路线图

**✅ 第一阶段："全栈算子锻造" - 已完成**
- [x] 核心张量数据结构与CPU参考实现
- [x] 所有基础算子的GPU核函数实现
- [x] 带多核函数编排的完整前馈神经网络
- [x] 全面的测试与验证框架
- [x] 模块化架构与专业工程实践
- [x] 技术文档化与知识总结

**🚧 第二阶段："高性能系统集成" - 即将开始**
- [ ] 高级GPU内存优化技术
- [ ] 核函数融合与计算图优化
- [ ] 多头注意力机制实现
- [ ] 完整GPT-2 Transformer块组装
- [ ] 性能分析与瓶颈识别

**🔮 第三阶段："生产就绪推理引擎" - 规划中**
- [ ] 预训练检查点模型加载 (GPT-2)
- [ ] 文本标记化与预处理流水线
- [ ] 批量推理优化
- [ ] 端到端文本生成能力
- [ ] 与生产框架的性能基准对比
