# CMake最低版本要求
cmake_minimum_required(VERSION 3.10)

# 项目名称
project(griffin)

# 指定头文件目录，这样 #include "utils.h" 才能找到文件
include_directories(include)

# 生成一个名为 "griffin" 的可执行文件
# 它由 src/main.cpp 和 src/utils.cpp 两个源文件编译而成
add_executable(griffin src/main.cpp src/utils.cpp)