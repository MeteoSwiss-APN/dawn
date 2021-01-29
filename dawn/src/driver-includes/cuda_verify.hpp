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
    if(expected == actual) {
    } else {
      error = fabs(expected - actual);
    }
    return error;
  }
};

template <>
struct compute_error<RelErrTag> {
  static double __device__ impl(const double expected, const double actual) {
    double error = 0.;
    if(expected == actual) {
    } else if(fabs(expected) < 1e-6 && fabs(actual) < 1e-6) {
      error = fabs(expected - actual);
    } else {
      error = fabs((expected - actual) / expected);
    }
    return error;
  }
};
} // namespace

namespace dawn {
template <typename error_type>
__global__ void compare_kernel(const int start_idx, const int end_idx, const int stride,
                               const int k_size, const double* __restrict__ dsl,
                               const double* __restrict__ fortran, double* __restrict__ error) {
  unsigned int eidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int kidx = blockIdx.y * blockDim.y + threadIdx.y;

  if(eidx < start_idx || eidx > end_idx) {
    error[kidx * blockDim.x * gridDim.x + eidx] = 0.;
    return;
  }

  if(kidx >= k_size) {
    error[kidx * blockDim.x * gridDim.x + eidx] = 0.;
    return;
  }

  error[kidx * blockDim.x * gridDim.x + eidx] =
      compute_error<error_type>::impl(fortran[kidx * stride + eidx], dsl[kidx * stride + eidx]);
}

// Returns relative error. Prints relative and absolute error.
inline double verify_field(const int start_idx, const int end_idx, const int stride,
                           const int k_size, const double* dsl, const double* actual,
                           std::string name) {
  double relErr, absErr;
  double* gpu_error;

  const int blockSize = 16;

  const int gridSizeX = ((end_idx + 1) + blockSize - 1) / blockSize,
            gridSizeY = (k_size + blockSize - 1) / blockSize;
  const int num_threads_tot = gridSizeX * gridSizeY * blockSize * blockSize;

  gpuErrchk(cudaMalloc((void**)&gpu_error, num_threads_tot * sizeof(double)));
  gpuErrchk(cudaPeekAtLastError());

  dim3 dGE(gridSizeX, gridSizeY, 1);
  dim3 dB(blockSize, blockSize, 1);

  gpuErrchk(cudaDeviceSynchronize());

  compare_kernel<RelErrTag>
      <<<dGE, dB>>>(start_idx, end_idx, stride, k_size, dsl, actual, gpu_error);
  gpuErrchk(cudaPeekAtLastError());

  thrust::device_ptr<double> dev_ptr;
  dev_ptr = thrust::device_pointer_cast(gpu_error);
  relErr = thrust::reduce(dev_ptr, dev_ptr + num_threads_tot, 0., thrust::maximum<double>());

  gpuErrchk(cudaPeekAtLastError());
  std::cout << "[DSL] " << name << " relative error: " << std::scientific << relErr << "\n"
            << std::flush;

  compare_kernel<AbsErrTag>
      <<<dGE, dB>>>(start_idx, end_idx, stride, k_size, dsl, actual, gpu_error);
  gpuErrchk(cudaPeekAtLastError());

  dev_ptr = thrust::device_pointer_cast(gpu_error);
  absErr = thrust::reduce(dev_ptr, dev_ptr + num_threads_tot, 0., thrust::maximum<double>());
  gpuErrchk(cudaPeekAtLastError());

  std::cout << "[DSL] " << name << " absolute error: " << std::scientific << absErr << "\n"
            << std::flush;
  gpuErrchk(cudaFree(gpu_error));
  gpuErrchk(cudaPeekAtLastError());

  return relErr;
}

} // namespace dawn
