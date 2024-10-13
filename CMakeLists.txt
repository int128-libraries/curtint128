cmake_minimum_required(VERSION 3.19 FATAL_ERROR) # for native CUDA support

project(cudauint128)

enable_language(CUDA)
set(CMAKE_CUDA_FLAGS "--expt-extended-lambda")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")

find_package(OpenMP REQUIRED)

add_executable(${PROJECT_NAME}_test_cuda src/test128cuda.cu)
target_include_directories(${PROJECT_NAME}_test_cuda PRIVATE include)
target_link_libraries(${PROJECT_NAME}_test_cuda)

add_executable(${PROJECT_NAME}_test_cpu src/test128cpu.cpp)
target_include_directories(${PROJECT_NAME}_test_cpu PRIVATE include)
target_link_libraries(${PROJECT_NAME}_test_cpu OpenMP::OpenMP_CXX)

