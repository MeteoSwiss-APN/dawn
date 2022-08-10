#include "cuda_verify.hpp"

namespace dawn {
__global__ void isclose_kernel(const int num_el, const double* __restrict__ dsl,
                               const double* __restrict__ fortran, bool* __restrict__ verified,
                               const double rel_tol, const double abs_tol) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(pidx >= num_el) {
    return;
  }

  // This verification method is inspired by numpy.isclose from Python
  verified[pidx] = (abs(dsl[pidx] - fortran[pidx]) <= (abs_tol + rel_tol * abs(fortran[pidx])));
}

VerificationMetrics verify_field(cudaStream_t stream, const int num_el, const double* dsl, const double* actual, std::string name,
                  const double rel_tol, const double abs_tol, const int iteration) {
  struct VerificationMetrics metrics;
  double* gpu_error;

  metrics.iteration = iteration;

  const int blockSize = 16;
  dim3 dG((num_el + blockSize - 1) / blockSize);
  dim3 dB(blockSize);

  gpuErrchk(cudaMalloc((void**)&gpu_error, num_el * sizeof(double)));
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  compare_kernel<RelErrTag><<<dG, dB, 0, stream>>>(num_el, dsl, actual, gpu_error);
  gpuErrchk(cudaPeekAtLastError());

  thrust::device_ptr<double> dev_ptr;
  dev_ptr = thrust::device_pointer_cast(gpu_error);

  metrics.maxRelErr = thrust::reduce(thrust::cuda::par.on(stream), dev_ptr, dev_ptr + num_el, 0., thrust::maximum<double>());
  gpuErrchk(cudaPeekAtLastError());

  std::cout << "[DSL] " << name << " maximum relative error: " << std::scientific << metrics.maxRelErr
            << "\n"
            << std::flush;

  metrics.minRelErr = thrust::reduce(thrust::cuda::par.on(stream), dev_ptr, dev_ptr + num_el, std::numeric_limits<double>::infinity(),
                             thrust::minimum<double>());
  gpuErrchk(cudaPeekAtLastError());

  std::cout << "[DSL] " << name << " minimum relative error: " << std::scientific << metrics.minRelErr
            << "\n"
            << std::flush;

  compare_kernel<AbsErrTag><<<dG, dB, 0, stream>>>(num_el, dsl, actual, gpu_error);
  gpuErrchk(cudaPeekAtLastError());

  dev_ptr = thrust::device_pointer_cast(gpu_error);
  metrics.maxAbsErr = thrust::reduce(thrust::cuda::par.on(stream), dev_ptr, dev_ptr + num_el, 0., thrust::maximum<double>());
  gpuErrchk(cudaPeekAtLastError());

  std::cout << "[DSL] " << name << " maximum absolute error: " << std::scientific << metrics.maxAbsErr
            << "\n"
            << std::flush;

  metrics.minAbsErr = thrust::reduce(thrust::cuda::par.on(stream), dev_ptr, dev_ptr + num_el, std::numeric_limits<double>::infinity(),
                             thrust::minimum<double>());
  gpuErrchk(cudaPeekAtLastError());

  std::cout << "[DSL] " << name << " minimum absolute error: " << std::scientific << metrics.minAbsErr
            << "\n"
            << std::flush;

  gpuErrchk(cudaFree(gpu_error));
  gpuErrchk(cudaPeekAtLastError());

  bool* gpu_verify_bools;

  gpuErrchk(cudaMalloc((void**)&gpu_verify_bools, num_el * sizeof(bool)));
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  isclose_kernel<<<dG, dB, 0, stream>>>(num_el, dsl, actual, gpu_verify_bools, rel_tol, abs_tol);
  gpuErrchk(cudaPeekAtLastError());

  thrust::device_ptr<bool> bool_dev_ptr;
  bool_dev_ptr = thrust::device_pointer_cast(gpu_verify_bools);
  gpuErrchk(cudaPeekAtLastError());

  metrics.isValid = thrust::all_of(thrust::cuda::par.on(stream), bool_dev_ptr, bool_dev_ptr + num_el, thrust::identity<bool>());

  if(!metrics.isValid) {
    double min, max, avg;
    thrust::device_ptr<const double> dev_cptr;

    dev_cptr = thrust::device_pointer_cast(actual);

    max = thrust::reduce(thrust::cuda::par.on(stream), dev_cptr, dev_cptr + num_el, -std::numeric_limits<double>::infinity(),
                         thrust::maximum<double>());
    gpuErrchk(cudaPeekAtLastError());
    min = thrust::reduce(thrust::cuda::par.on(stream), dev_cptr, dev_cptr + num_el, std::numeric_limits<double>::infinity(),
                         thrust::minimum<double>());
    gpuErrchk(cudaPeekAtLastError());
    avg = thrust::reduce(thrust::cuda::par.on(stream), dev_cptr, dev_cptr + num_el) / num_el;
    gpuErrchk(cudaPeekAtLastError());

    std::cout << "[DSL] " << name << " max: " << max << "\n" << std::flush;
    std::cout << "[DSL] " << name << " min: " << min << "\n" << std::flush;
    std::cout << "[DSL] " << name << " avg: " << avg << "\n" << std::flush;

    dev_cptr = thrust::device_pointer_cast(dsl);

    max = thrust::reduce(thrust::cuda::par.on(stream), dev_cptr, dev_cptr + num_el, -std::numeric_limits<double>::infinity(),
                         thrust::maximum<double>());
    gpuErrchk(cudaPeekAtLastError());
    min = thrust::reduce(thrust::cuda::par.on(stream), dev_cptr, dev_cptr + num_el, std::numeric_limits<double>::infinity(),
                         thrust::minimum<double>());
    gpuErrchk(cudaPeekAtLastError());
    avg = thrust::reduce(thrust::cuda::par.on(stream), dev_cptr, dev_cptr + num_el) / num_el;
    gpuErrchk(cudaPeekAtLastError());

    std::cout << "[DSL] " << name << "_dsl max: " << max << "\n" << std::flush;
    std::cout << "[DSL] " << name << "_dsl min: " << min << "\n" << std::flush;
    std::cout << "[DSL] " << name << "_dsl avg: " << avg << "\n" << std::flush;
  }

  gpuErrchk(cudaFree(gpu_verify_bools));
  gpuErrchk(cudaPeekAtLastError());
  return metrics;
}

} // namespace dawn
