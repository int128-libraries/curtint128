# A 128 bit unsigned integer class for CUDA

This library seeks to provide a convenient `uint128_t` type for use in CUDA code, as well as in C++. At this point, nearly all operators have been overloaded for the `uint128_t` class, all common arithmetic and bit arithmetic functions (including several roots) are defined. Additionally, more functions have been defined to increase the usability of this class, including typecasts to and from float and double, ostream insert (on the host), and `std::string` -> `uint128_t` conversion.

## Usage

The `cuda_uint128.h` header can be seamlessly `included` into both `.cu` and `.cpp` source files. Due to inefficiencies in linking device code with nvcc, this is a header-only library.

## Testing

C++ and CUDA test cases are provided in `src` and could be built with CMake:

```
mkdir build
cd build
cmake ..
make
./cudauint128_test_cuda
./cudauint128_test_cpu
```

