cmake_minimum_required(VERSION 3.18)
project(CUTLASS_MLP LANGUAGES CXX CUDA)

# 设置 CUTLASS 的路径
set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/dependencies/cutlass")

# 查找 CUDA 包
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 包含 CUTLASS 头文件
include_directories(${CUTLASS_DIR}/include ${CUTLASS_DIR}/tools/util/include)

# 添加源文件
file(GLOB SOURCES "src/*.cpp" "src/*.cu")
include_directories(${CMAKE_SOURCE_DIR}/include)

# 指定使用 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 定义编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")

# 设置可执行文件输出路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# 定义可执行文件
add_executable(app ${SOURCES})

# # 链接 CUTLASS 库
# target_link_libraries(my_cutlass_project ${CUTLASS_DIR}/build/lib/libcutlass.a)