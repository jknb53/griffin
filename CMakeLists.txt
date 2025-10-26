
 cmake_minimum_required(VERSION 3.10)
 project(griffin LANGUAGES CXX CUDA)

 # 第1步: 找到CUDA工具包。这是所有CUDA相关设置的前提。
 # find_package 命令会创建一些变量，比如 CUDA_INCLUDE_DIRS 和 CUDA_LIBRARIES
 find_package(CUDA REQUIRED)

 # 第2步: 定义我们的可执行文件目标 "griffin"
 # 它由哪些源文件构成
 add_executable(griffin
     src/main.cpp
     src/cpu_ops.cpp
     src/kernel.cu
 )

 # 第3步: 为 "griffin" 这个目标配置它所需要的东西

 # 告诉 griffin 去哪里寻找我们自己的头文件 (utils.h)
 target_include_directories(griffin PUBLIC
     ${CMAKE_CURRENT_SOURCE_DIR}/include
 )

 # 【关键修复】告诉 griffin 去哪里寻找CUDA的官方头文件 (cuda_runtime.h)
 # 我们使用 find_package 创建的 CUDA_INCLUDE_DIRS 变量
 target_include_directories(griffin PUBLIC
     ${CUDA_INCLUDE_DIRS}
 )

 # 【关键修复】告诉 griffin 它需要链接哪些CUDA的官方库文件
 # 我们使用 find_package 创建的 CUDA_LIBRARIES 变量
 target_link_libraries(griffin PUBLIC
     ${CUDA_LIBRARIES}
 )
 # --- 【替换结束】 ---
 