#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/to_vtk.h"
#define GRIDTOOLS_DAWN_NO_INCLUDE
#include "driver-includes/math.hpp"
#include "driver-includes/timer_cuda.hpp"
#include <chrono>
#define BLOCK_SIZE 16
#define LEVELS_PER_THREAD 1
using namespace gridtools::dawn;

namespace dawn_generated {
namespace cuda_ico {
template <int E_C_SIZE, int E_C_E_SIZE>
__global__ void offset_reduction_cuda_stencil34_ms47_s52_kernel(
    int EdgeStride, int kSize, int hSize, const int* ecTable, const int* eceTable,
    ::dawn::float_type* __restrict__ out_vn_e,
    const ::dawn::float_type* __restrict__ raw_diam_coeff,
    const ::dawn::float_type* __restrict__ prism_thick_e,
    const ::dawn::float_type* __restrict__ e2c_aux,
    const ::dawn::float_type* __restrict__ e2c_aux_h) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int kidx = blockIdx.y * blockDim.y + threadIdx.y;
  int klo = kidx * LEVELS_PER_THREAD + 0;
  int khi = (kidx + 1) * LEVELS_PER_THREAD + 0;
  if(pidx >= hSize) {
    return;
  }
  for(int kIter = klo; kIter < khi; kIter++) {
    if(kIter >= kSize + 0) {
      return;
    }
    ::dawn::float_type lhs_36 = (::dawn::float_type).0;
    ::dawn::float_type weights_36[4] = {
        e2c_aux[0 * kSize * EdgeStride + (kIter + 0) * EdgeStride + pidx],
        e2c_aux[0 * kSize * EdgeStride + (kIter + 0) * EdgeStride + pidx],
        e2c_aux[1 * kSize * EdgeStride + (kIter + 0) * EdgeStride + pidx],
        e2c_aux[1 * kSize * EdgeStride + (kIter + 0) * EdgeStride + pidx]};
    for(int nbhIter = 0; nbhIter < E_C_E_SIZE; nbhIter++) {
      int nbhIdx = eceTable[pidx * E_C_E_SIZE + nbhIter];
      if(nbhIdx == DEVICE_MISSING_VALUE) {
        continue;
      }
      lhs_36 += weights_36[nbhIter] *
                (raw_diam_coeff[nbhIter * kSize * EdgeStride + (kIter + 0) * EdgeStride + pidx] *
                 prism_thick_e[(kIter + 0) * EdgeStride + nbhIdx]);
    }
    out_vn_e[(kIter + 0) * EdgeStride + pidx] = lhs_36;
    ::dawn::float_type lhs_40 = (::dawn::float_type).0;
    ::dawn::float_type weights_40[4] = {
        e2c_aux_h[0 * EdgeStride + pidx], e2c_aux_h[0 * EdgeStride + pidx],
        e2c_aux_h[1 * EdgeStride + pidx], e2c_aux_h[1 * EdgeStride + pidx]};
    for(int nbhIter = 0; nbhIter < E_C_E_SIZE; nbhIter++) {
      int nbhIdx = eceTable[pidx * E_C_E_SIZE + nbhIter];
      if(nbhIdx == DEVICE_MISSING_VALUE) {
        continue;
      }
      lhs_40 += weights_40[nbhIter] *
                (raw_diam_coeff[nbhIter * kSize * EdgeStride + (kIter + 0) * EdgeStride + pidx] *
                 prism_thick_e[(kIter + 0) * EdgeStride + nbhIdx]);
    }
    out_vn_e[(kIter + 0) * EdgeStride + pidx] = lhs_40;
  }
}

class offset_reduction_cuda {
public:
  static const int E_C_E_SIZE = 4;
  static const int E_C_SIZE = 2;

  struct sbase : public timer_cuda {

    sbase(std::string name) : timer_cuda(name) {}

    double get_time() { return total_time(); }
  };

  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    dawn::unstructured_domain Domain;
    int* eceTable;
    int* ecTable;

    GpuTriMesh() {}

    GpuTriMesh(const dawn::GlobalGpuTriMesh* mesh) {
      NumVertices = mesh->NumVertices;
      NumCells = mesh->NumCells;
      NumEdges = mesh->NumEdges;
      Domain = mesh->Domain;
      eceTable = mesh->NeighborTables.at(std::tuple<std::vector<dawn::LocationType>, bool>{
          {dawn::LocationType::Edges, dawn::LocationType::Cells, dawn::LocationType::Edges}, 0});
      ecTable = mesh->NeighborTables.at(std::tuple<std::vector<dawn::LocationType>, bool>{
          {dawn::LocationType::Edges, dawn::LocationType::Cells}, 0});
    }
  };

  struct stencil_34 : public sbase {
  private:
    ::dawn::float_type* out_vn_e_;
    ::dawn::float_type* raw_diam_coeff_;
    ::dawn::float_type* prism_thick_e_;
    ::dawn::float_type* e2c_aux_;
    ::dawn::float_type* e2c_aux_h_;
    static int kSize_;
    static GpuTriMesh mesh_;
    static bool is_setup_;

  public:
    static const GpuTriMesh& getMesh() { return mesh_; }

    static int getKSize() { return kSize_; }

    static void free() {}

    static void setup(const dawn::GlobalGpuTriMesh* mesh, int kSize) {
      mesh_ = GpuTriMesh(mesh);
      kSize_ = kSize;
      is_setup_ = true;
    }

    dim3 grid(int kSize, int elSize) {
      int dK = (kSize + LEVELS_PER_THREAD - 1) / LEVELS_PER_THREAD;
      return dim3((elSize + BLOCK_SIZE - 1) / BLOCK_SIZE, (dK + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    }

    stencil_34() : sbase("stencil_34") {}

    void run() {
      if(!is_setup_) {
        printf(
            "offset_reduction_cuda has not been set up! make sure setup() is called before run!\n");
        return;
      }
      dim3 dB(BLOCK_SIZE, BLOCK_SIZE, 1);
      sbase::start();
      int hsize52 = mesh_.NumEdges;
      dim3 dG52 = grid(kSize_ + 0 - 0, hsize52);
      offset_reduction_cuda_stencil34_ms47_s52_kernel<E_C_SIZE, E_C_E_SIZE>
          <<<dG52, dB>>>(mesh_.NumEdges + 0, kSize_, hsize52, mesh_.ecTable, mesh_.eceTable,
                         out_vn_e_, raw_diam_coeff_, prism_thick_e_, e2c_aux_, e2c_aux_h_);
#ifndef NDEBUG

      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
#endif

      sbase::pause();
    }

    void CopyResultToHost(::dawn::float_type* out_vn_e, bool do_reshape) {
      if(do_reshape) {
        ::dawn::float_type* host_buf = new ::dawn::float_type[(mesh_.NumEdges + 0) * kSize_];
        gpuErrchk(cudaMemcpy((::dawn::float_type*)host_buf, out_vn_e_,
                             (mesh_.NumEdges + 0) * kSize_ * sizeof(::dawn::float_type),
                             cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, out_vn_e, kSize_, mesh_.NumEdges + 0);
        delete[] host_buf;
      } else {
        gpuErrchk(cudaMemcpy(out_vn_e, out_vn_e_,
                             (mesh_.NumEdges + 0) * kSize_ * sizeof(::dawn::float_type),
                             cudaMemcpyDeviceToHost));
      }
    }

    void copy_memory(::dawn::float_type* out_vn_e, ::dawn::float_type* raw_diam_coeff,
                     ::dawn::float_type* prism_thick_e, ::dawn::float_type* e2c_aux,
                     ::dawn::float_type* e2c_aux_h, bool do_reshape) {
      dawn::initField(out_vn_e, &out_vn_e_, mesh_.NumEdges + 0, kSize_, do_reshape);
      dawn::initSparseField(raw_diam_coeff, &raw_diam_coeff_, mesh_.NumEdges + 0, E_C_E_SIZE,
                            kSize_, do_reshape);
      dawn::initField(prism_thick_e, &prism_thick_e_, mesh_.NumEdges + 0, kSize_, do_reshape);
      dawn::initSparseField(e2c_aux, &e2c_aux_, mesh_.NumEdges + 0, E_C_SIZE, kSize_, do_reshape);
      dawn::initSparseField(e2c_aux_h, &e2c_aux_h_, mesh_.NumEdges + 0, E_C_SIZE, 1, do_reshape);
    }

    void copy_pointers(::dawn::float_type* out_vn_e, ::dawn::float_type* raw_diam_coeff,
                       ::dawn::float_type* prism_thick_e, ::dawn::float_type* e2c_aux,
                       ::dawn::float_type* e2c_aux_h) {
      out_vn_e_ = out_vn_e;
      raw_diam_coeff_ = raw_diam_coeff;
      prism_thick_e_ = prism_thick_e;
      e2c_aux_ = e2c_aux;
      e2c_aux_h_ = e2c_aux_h;
    }
  };
};
} // namespace cuda_ico
} // namespace dawn_generated
extern "C" {
double run_offset_reduction_cuda_from_c_host(dawn::GlobalGpuTriMesh* mesh, int k_size,
                                             ::dawn::float_type* out_vn_e,
                                             ::dawn::float_type* raw_diam_coeff,
                                             ::dawn::float_type* prism_thick_e,
                                             ::dawn::float_type* e2c_aux,
                                             ::dawn::float_type* e2c_aux_h) {
  dawn_generated::cuda_ico::offset_reduction_cuda::stencil_34 s;
  dawn_generated::cuda_ico::offset_reduction_cuda::stencil_34::setup(mesh, k_size);
  s.copy_memory(out_vn_e, raw_diam_coeff, prism_thick_e, e2c_aux, e2c_aux_h, true);
  s.run();
  double time = s.get_time();
  s.reset();
  s.CopyResultToHost(out_vn_e, true);
  dawn_generated::cuda_ico::offset_reduction_cuda::stencil_34::free();
  return time;
}
double run_offset_reduction_cuda_from_fort_host(dawn::GlobalGpuTriMesh* mesh, int k_size,
                                                ::dawn::float_type* out_vn_e,
                                                ::dawn::float_type* raw_diam_coeff,
                                                ::dawn::float_type* prism_thick_e,
                                                ::dawn::float_type* e2c_aux,
                                                ::dawn::float_type* e2c_aux_h) {
  dawn_generated::cuda_ico::offset_reduction_cuda::stencil_34 s;
  dawn_generated::cuda_ico::offset_reduction_cuda::stencil_34::setup(mesh, k_size);
  s.copy_memory(out_vn_e, raw_diam_coeff, prism_thick_e, e2c_aux, e2c_aux_h, false);
  s.run();
  double time = s.get_time();
  s.reset();
  s.CopyResultToHost(out_vn_e, false);
  dawn_generated::cuda_ico::offset_reduction_cuda::stencil_34::free();
  return time;
}
double run_offset_reduction_cuda(::dawn::float_type* out_vn_e, ::dawn::float_type* raw_diam_coeff,
                                 ::dawn::float_type* prism_thick_e, ::dawn::float_type* e2c_aux,
                                 ::dawn::float_type* e2c_aux_h) {
  dawn_generated::cuda_ico::offset_reduction_cuda::stencil_34 s;
  s.copy_pointers(out_vn_e, raw_diam_coeff, prism_thick_e, e2c_aux, e2c_aux_h);
  s.run();
  double time = s.get_time();
  s.reset();
  return time;
}
bool verify_offset_reduction_cuda(const ::dawn::float_type* out_vn_e_dsl,
                                  const ::dawn::float_type* out_vn_e, const int iteration,
                                  const double rel_err_threshold) {
  using namespace std::chrono;
  const auto& mesh = dawn_generated::cuda_ico::offset_reduction_cuda::stencil_34::getMesh();
  int kSize = dawn_generated::cuda_ico::offset_reduction_cuda::stencil_34::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  bool isValid = true;
  double relErr;
  relErr = ::dawn::verify_field((mesh.NumEdges + 0) * kSize, out_vn_e_dsl, out_vn_e, "out_vn_e");
  if(relErr > rel_err_threshold) {
    isValid = false;
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(0, (mesh.NumEdges - 1), kSize, (mesh.NumEdges + 0), out_vn_e,
                          "offset_reduction_cuda", "out_vn_e", iteration);
    serialize_dense_edges(0, (mesh.NumEdges - 1), kSize, (mesh.NumEdges + 0), out_vn_e_dsl,
                          "offset_reduction_cuda", "out_vn_e_dsl", iteration);
    std::cout << "[DSL] serializing out_vn_e as error is high.\n" << std::flush;
#endif
  }
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n" << std::flush;
  return isValid;
}
void run_and_verify_offset_reduction_cuda(
    ::dawn::float_type* out_vn_e, ::dawn::float_type* raw_diam_coeff,
    ::dawn::float_type* prism_thick_e, ::dawn::float_type* e2c_aux, ::dawn::float_type* e2c_aux_h,
    ::dawn::float_type* out_vn_e_before, const double rel_err_threshold) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil offset_reduction_cuda...\n" << std::flush;
  double time =
      run_offset_reduction_cuda(out_vn_e_before, raw_diam_coeff, prism_thick_e, e2c_aux, e2c_aux_h);
  std::cout << "[DSL] offset_reduction_cuda run time: " << time << "s\n" << std::flush;
  std::cout << "[DSL] Verifying stencil offset_reduction_cuda...\n" << std::flush;
  verify_offset_reduction_cuda(out_vn_e_before, out_vn_e, iteration, rel_err_threshold);
  iteration++;
}
void setup_offset_reduction_cuda(dawn::GlobalGpuTriMesh* mesh, int k_size) {
  dawn_generated::cuda_ico::offset_reduction_cuda::stencil_34::setup(mesh, k_size);
}
void free_offset_reduction_cuda() {
  dawn_generated::cuda_ico::offset_reduction_cuda::stencil_34::free();
}
}
int dawn_generated::cuda_ico::offset_reduction_cuda::stencil_34::kSize_;
bool dawn_generated::cuda_ico::offset_reduction_cuda::stencil_34::is_setup_ = false;
dawn_generated::cuda_ico::offset_reduction_cuda::GpuTriMesh
    dawn_generated::cuda_ico::offset_reduction_cuda::stencil_34::mesh_;
