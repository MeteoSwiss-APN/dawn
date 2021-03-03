#pragma once

#include "cuda_utils.hpp"

#include <iostream>

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

namespace {

struct AbsErrTag {};
struct RelErrTag {};

template <typename error_type>
struct compute_error {
  static double __device__ impl(const double expected, const double actual);
};

template <>
struct compute_error<AbsErrTag> {
  static double __device__ impl(const double expected, const double actual) {
    double error = 0.;
    if(expected != actual) {
      error = fabs(expected - actual);
    }
    return error;
  }
};

template <>
struct compute_error<RelErrTag> {
  static double __device__ impl(const double expected, const double actual) {
    double error = 0.;
    if(expected != actual) {
      error = fabs((expected - actual) / expected);
    }
    return error;
  }
};
} // namespace

namespace dawn {
template <typename error_type>
__global__ void compare_kernel(const int num_el, const double* __restrict__ dsl,
                               const double* __restrict__ fortran, double* __restrict__ error) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(pidx >= num_el) {
    return;
  }

  error[pidx] = compute_error<error_type>::impl(fortran[pidx], dsl[pidx]);
}

// Returns relative error. Prints relative and absolute error.
inline double verify_field(const int num_el, const double* dsl, const double* actual,
                           std::string name) {
  double relErr, absErr;
  double* gpu_error;

  const int blockSize = 16;
  dim3 dG((num_el + blockSize - 1) / blockSize);
  dim3 dB(blockSize);

  gpuErrchk(cudaMalloc((void**)&gpu_error, num_el * sizeof(double)));
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  compare_kernel<RelErrTag><<<dG, dB>>>(num_el, dsl, actual, gpu_error);
  gpuErrchk(cudaPeekAtLastError());

  thrust::device_ptr<double> dev_ptr;
  dev_ptr = thrust::device_pointer_cast(gpu_error);
  relErr = thrust::reduce(dev_ptr, dev_ptr + num_el, 0., thrust::maximum<double>());

  gpuErrchk(cudaPeekAtLastError());
  std::cout << "[DSL] " << name << " relative error: " << std::scientific << relErr << "\n"
            << std::flush;

  compare_kernel<AbsErrTag><<<dG, dB>>>(num_el, dsl, actual, gpu_error);
  gpuErrchk(cudaPeekAtLastError());

  dev_ptr = thrust::device_pointer_cast(gpu_error);
  absErr = thrust::reduce(dev_ptr, dev_ptr + num_el, 0., thrust::maximum<double>());
  gpuErrchk(cudaPeekAtLastError());

  std::cout << "[DSL] " << name << " absolute error: " << std::scientific << absErr << "\n"
            << std::flush;
  gpuErrchk(cudaFree(gpu_error));
  gpuErrchk(cudaPeekAtLastError());

  return relErr;
}

} // namespace dawn
