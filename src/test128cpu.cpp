#include <cstdio>
#include <cstdint>
#include <gtest/gtest.h>

#include "cuda_uint128.h"

#if (defined __GNUC__ || defined __clang__) && defined __SIZEOF_INT128__
#define HAS_NATIVE_UINT128_T 1
#else
#undef HAS_NATIVE_UINT128_T
#endif

using uint128_t = uint128_t;

static void Test(std::uint64_t x) {
  uint128_t n{x};
  EXPECT_EQ(x, static_cast<std::uint64_t>(n));
  EXPECT_EQ(~x, static_cast<std::uint64_t>(~n));
  EXPECT_EQ(-x, static_cast<std::uint64_t>(-n));
  EXPECT_EQ(!x, static_cast<std::uint64_t>(!n));
  EXPECT_TRUE(n == n);
  EXPECT_TRUE(n + n == n * static_cast<uint128_t>(2));
  EXPECT_TRUE(n - n == static_cast<uint128_t>(0));
  EXPECT_TRUE(n + n == n << static_cast<uint128_t>(1));
  EXPECT_TRUE(n + n == n << static_cast<uint128_t>(1));
  EXPECT_TRUE((n + n) - n == n);
  EXPECT_TRUE(((n + n) >> static_cast<uint128_t>(1)) == n);
  if (x != 0) {
    EXPECT_TRUE(static_cast<uint128_t>(0) / n == static_cast<uint128_t>(0));
    EXPECT_TRUE(static_cast<uint128_t>(n - 1) / n == static_cast<uint128_t>(0));
    EXPECT_TRUE(static_cast<uint128_t>(n) / n == static_cast<uint128_t>(1));
    EXPECT_TRUE(static_cast<uint128_t>(n + n - 1) / n == static_cast<uint128_t>(1));
    EXPECT_TRUE(static_cast<uint128_t>(n + n) / n == static_cast<uint128_t>(2));
  }
}

static void Test(std::uint64_t x, std::uint64_t y) {
  uint128_t m{x}, n{y};
  EXPECT_EQ(x, static_cast<std::uint64_t>(m));
  EXPECT_EQ(y, static_cast<std::uint64_t>(n));
  EXPECT_EQ(x & y, static_cast<std::uint64_t>(m & n));
  EXPECT_EQ(x | y, static_cast<std::uint64_t>(m | n));
  EXPECT_EQ(x ^ y, static_cast<std::uint64_t>(m ^ n));
  EXPECT_EQ(x + y, static_cast<std::uint64_t>(m + n));
  EXPECT_EQ(x - y, static_cast<std::uint64_t>(m - n));
  EXPECT_EQ(x * y, static_cast<std::uint64_t>(m * n));
  if (n != 0) {
    EXPECT_EQ(x / y, static_cast<std::uint64_t>(m / n));
  }
}

#if HAS_NATIVE_UINT128_T
static __uint128_t ToNative(uint128_t n) {
  return static_cast<__uint128_t>(static_cast<std::uint64_t>(n >> 64)) << 64 |
      static_cast<std::uint64_t>(n);
}

static uint128_t FromNative(__uint128_t n) {
  return uint128_t{static_cast<std::uint64_t>(n >> 64)} << 64 |
      uint128_t{static_cast<std::uint64_t>(n)};
}

static void TestVsNative(__uint128_t x, __uint128_t y) {
  uint128_t m{FromNative(x)}, n{FromNative(y)};
  EXPECT_TRUE(ToNative(m) == x);
  EXPECT_TRUE(ToNative(n) == y);
  EXPECT_TRUE(ToNative(~m) == ~x);
  EXPECT_TRUE(ToNative(-m) == -x);
  EXPECT_TRUE(ToNative(!m) == !x);
  EXPECT_TRUE(ToNative(m < n) == (x < y));
  EXPECT_TRUE(ToNative(m <= n) == (x <= y));
  EXPECT_TRUE(ToNative(m == n) == (x == y));
  EXPECT_TRUE(ToNative(m != n) == (x != y));
  EXPECT_TRUE(ToNative(m >= n) == (x >= y));
  EXPECT_TRUE(ToNative(m > n) == (x > y));
  EXPECT_TRUE(ToNative(m & n) == (x & y));
  EXPECT_TRUE(ToNative(m | n) == (x | y));
  EXPECT_TRUE(ToNative(m ^ n) == (x ^ y));
  if (y < 128) {
    EXPECT_TRUE(ToNative(m << n) == (x << y));
    EXPECT_TRUE(ToNative(m >> n) == (x >> y));
  }
  EXPECT_TRUE(ToNative(m + n) == (x + y));
  EXPECT_TRUE(ToNative(m - n) == (x - y));
  EXPECT_TRUE(ToNative(m * n) == (x * y));
  if (y > 0) {
    EXPECT_TRUE(ToNative(m / n) == (x / y));
    EXPECT_TRUE(ToNative(m % n) == (x % y));
    EXPECT_TRUE(ToNative(m - n * (m / n)) == (x % y));
  }
}

static void TestVsNative() {
  for (int j{0}; j < 128; ++j) {
    for (int k{0}; k < 128; ++k) {
      __uint128_t m{1}, n{1};
      m <<= j, n <<= k;
      TestVsNative(m, n);
      TestVsNative(~m, n);
      TestVsNative(m, ~n);
      TestVsNative(~m, ~n);
      TestVsNative(m ^ n, n);
      TestVsNative(m, m ^ n);
      TestVsNative(m ^ ~n, n);
      TestVsNative(m, ~m ^ n);
      TestVsNative(m ^ ~n, m ^ n);
      TestVsNative(m ^ n, ~m ^ n);
      TestVsNative(m ^ ~n, ~m ^ n);
      Test(m, 10000000000000000); // important case for decimal conversion
      Test(~m, 10000000000000000);
    }
  }
}
#endif

TEST(uint128, Test1) {
  for (std::uint64_t j{0}; j < 64; ++j) {
    Test(j);
    Test(~j);
    Test(std::uint64_t(1) << j);
    for (std::uint64_t k{0}; k < 64; ++k) {
      Test(j, k);
    }
  }
#if HAS_NATIVE_UINT128_T
  TestVsNative();
#else
  fprintf(stderr, "Environment lacks native __uint128_t\n");
#endif
}

TEST(uint128, Test2) {
  uint128_t x = (uint128_t) 1 << 120;

  #pragma omp parallel for
  for (uint64_t v = 2; v < 1u << 30; v++) {
    uint64_t r;
    uint128_t y = uint128_t::div128to128(x, v, &r);
    uint128_t z = mul128(y, v) + r;

    if (z != x)
      fprintf(stderr, "Error : y (%s) * v (%lu) + r (%lu) != x (%s)\n",
        u128_to_string(y).c_str(), v, r, u128_to_string(x).c_str());
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

