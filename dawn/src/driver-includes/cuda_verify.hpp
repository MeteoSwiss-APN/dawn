#pragma once

#include "cuda_utils.hpp"

#include <iostream>

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>

namespace {

struct AbsErrTag {};
struct RelErrTag {};


struct VerificationMetrics {
  bool isValid;
  double maxRelErr;
  double minRelErr;
  double maxAbsErr;
  double minAbsErr;
}

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

VerificationMetrics verify_field(cudaStream_t stream, const int num_el, const double* dsl, const double* actual, std::string name,
                  const double rel_tol, const double abs_tol);

} // namespace dawn
