#include <cstdio>
#include <stdint.h>
#include <iostream>

#include "cuda_uint128.h"

int main(int argc, char ** argv)
{
  uint128_t x = (uint128_t) 1 << 120;

  if (argc == 2)
    x = string_to_u128((std::string)argv[1]);

  #pragma omp parallel for
  for (uint64_t v = 2; v < 1u << 30; v++) {
    uint64_t r;
    uint128_t y = uint128_t::div128to128(x, v, &r);
    uint128_t z = mul128(y, v) + r;

    if (z != x)
      fprintf(stderr, "Error : y (%s) * v (%lu) + r (%lu) != x (%s)\n",
        u128_to_string(y).c_str(), v, r, u128_to_string(x).c_str());
  }

  printf("Done!\n");

  return 0;
}
