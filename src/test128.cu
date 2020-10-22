#include <iostream>
#include <stdint.h>
#include <math.h>
#include <string>
#include <curand.h>

#include <CUDASieve/cudasieve.hpp>
#include <CUDASieve/host.hpp>

#include "cuda_uint128.h"
#include "cuda_uint128_primitives.cuh"
#include "utils.h"

uint64_t * generateUniform64(uint64_t num);

__global__
void atimesbequalsc(uint64_t * a, uint64_t * b, uint128_t * c);
__global__
void squarerootc(uint128_t * c, uint64_t * a);
__global__
void sqrt_test(uint64_t * a, volatile uint64_t * errors);
__global__
void div_test(uint64_t * a, volatile uint64_t * errors);


int main(int argc, char * argv[])
{
  uint128_t x = (uint128_t) 1 << 120;

  if(argc == 2)
    x = string_to_u128((std::string)argv[1]);

  #pragma omp parallel for
  for(uint64_t v = 2; v < 1u << 30; v++){
    uint64_t r;
    uint128_t y = uint128_t::div128to128(x, v, &r);
    uint128_t z = mul128(y, v) + r;

    if(z != x) std::cout << z << std::endl;
  }

  // uint64_t * d64 = generateUniform64(1u<<26);
  // volatile uint64_t * h_errors, * d_errors;
  // cudaHostAlloc((void **)&h_errors, sizeof(uint64_t), cudaHostAllocMapped);
  // cudaHostGetDevicePointer((uint64_t **)&d_errors, (uint64_t *)h_errors, 0);
  //
  // *h_errors = 0;
  //
  // KernelTime timer;
  //
  // timer.start();
  //
  // div_test<<<65536, 256>>>(d64, d_errors);
  //
  // cudaDeviceSynchronize();
  // timer.stop();
  // timer.displayTime();
  //
  // std::cout << *h_errors << " errors " << std::endl;

  return 0;
}

uint64_t * generateUniform64(uint64_t num)
{
  uint64_t * d_r;
  curandGenerator_t gen;

  cudaMalloc(&d_r, num * sizeof(uint64_t));

  curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL64);
  curandSetPseudoRandomGeneratorSeed(gen, 1278459ull);
  curandGenerateLongLong(gen, (unsigned long long *)d_r, num);

  return d_r;
}

__global__
void atimesbequalsc(uint64_t * a, uint64_t * b, uint128_t * c)
{
  uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
  c[tidx] = mul128(a[tidx], b[tidx]);
}

__global__
void squarerootc(uint128_t * c, uint64_t * a)
{
  uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
  a[tidx] = _isqrt(c[tidx]);
  if(mul128(a[tidx], a[tidx]) > c[tidx] || mul128((a[tidx] + 1), (a[tidx] + 1)) <= c[tidx])
    printf("%llu  %f  %llu\n", a[tidx], u128_to_float(c[tidx]), c[tidx].hi);
}

__global__
void sqrt_test(uint64_t * a, volatile uint64_t * errors)
{
  __shared__ uint64_t s_a[1024];

  #pragma unroll
  for(uint16_t i = 0; i < 4; i++){
    s_a[threadIdx.x + i * blockDim.x] = a[threadIdx.x + i * blockDim.x + 1024*blockIdx.x];
  }
  __syncthreads();

  uint128_t x;
  #pragma unroll
  for(uint16_t i = 0; i < 4; i++){
    x.lo = s_a[threadIdx.x + i * blockDim.x];
    #pragma unroll
    for(uint16_t i = 0; i < 1024; i++){
      x.hi = s_a[i] >> 4;
      uint64_t y = _isqrt(x);
      if(mul128(y,y) > x || mul128(y + 1, y + 1) <= x){
        atomicAdd((unsigned long long *)errors, 1ull);
        printf("%llu %llu %llu\n", x.hi, x.lo, y);
      }
    }
  }
}

__global__
void div_test(uint64_t * a, volatile uint64_t * errors)
{
  __shared__ uint64_t s_a[1024];

  #pragma unroll
  for(uint16_t i = 0; i < 4; i++){
    s_a[threadIdx.x + i * blockDim.x] = a[threadIdx.x + i * blockDim.x + 1024*blockIdx.x];
  }
  __syncthreads();

  uint128_t x, y;
  uint64_t v, r;
  #pragma unroll
  for(uint16_t i = 0; i < 4; i++){
    x.lo = s_a[threadIdx.x + i * blockDim.x];
    #pragma unroll
    for(uint16_t i = 0; i < 1024; i++){
      x.hi = s_a[i] >> 4;
      v = s_a[(i + 1 )& 1023] >> (x.hi & 31);
      y = div128to128(x,v,&r);
      y = add128(mul128(y, v), r);
      uint64_t y = _isqrt(x);
      // if(y != x){
      //   atomicAdd((unsigned long long *)errors, 1ull);
      // }
    }
  }
}
