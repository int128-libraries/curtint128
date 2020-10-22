# Location of the CUDA toolkit
CUDA_DIR = /opt/cuda
# Compute capability of the target GPU
GPU_ARCH = compute_60
GPU_CODE = sm_60


# Compilers to use
NVCC = nvcc
LEGACY_CC_PATH = g++
# Flags for the host compiler
CCFLAGS = -O3 -std=c++11 -c
WIGNORE = -Wno-return-stack-address

# Flags for nvcc
# ptxas-options=-dlcm=cg (vs. default of ca) is about a 2% performance gain
NVCC_FLAGS = -ccbin $(LEGACY_CC_PATH) -std=c++11 -arch=$(GPU_ARCH) -code=$(GPU_CODE)
TEST = test
TEST_CPU = testcpu
SRC = test128.cu
SRC_CPU = test128cpu.cu
INCLUDE = cuda_uint128.h
INCLUDE_PATHS = -I /home/curtis/Projects/CUDASieve/include
CUDASIEVE_LIB = /home/curtis/Projects/CUDASieve/libcudasieve.a
LIBS = -lcurand

$(TEST): $(SRC) $(INCLUDE)
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDE_PATHS) $(CUDASIEVE_LIB) $(LIBS) $< -o $@
	@echo "     CUDA     " $@

$(TEST_CPU) : $(SRC_CPU) $(INCLUDE)
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDE_PATHS) -Xcompiler=-fopenmp $< -o $@
