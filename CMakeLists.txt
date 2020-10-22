cmake_minimum_required(VERSION 3.11)

project(cudauint128)

enable_language(CUDA)

find_package(OpenMP REQUIRED)

file(GLOB SIEVE_SRCS "ThirdParty/CUDASieve/src/CUDASieve/*.cpp" "ThirdParty/CUDASieve/src/CUDASieve/*.cu")
add_library(cudasieve ${SIEVE_SRCS})
set_target_properties(cudasieve PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(cudasieve PRIVATE ThirdParty/CUDASieve/include)

add_executable(${PROJECT_NAME}_test_cuda cuda_uint128.h cuda_uint128_primitives.cuh test128.cu)
target_include_directories(${PROJECT_NAME}_test_cuda PRIVATE ThirdParty/CUDASieve/include)
target_link_libraries(${PROJECT_NAME}_test_cuda cudasieve curand)

add_executable(${PROJECT_NAME}_test_cpu cuda_uint128.h cuda_uint128_primitives.cuh test128cpu.cu)
target_include_directories(${PROJECT_NAME}_test_cpu PRIVATE ThirdParty/CUDASieve/include)
target_link_libraries(${PROJECT_NAME}_test_cpu cudasieve OpenMP::OpenMP_CXX)

