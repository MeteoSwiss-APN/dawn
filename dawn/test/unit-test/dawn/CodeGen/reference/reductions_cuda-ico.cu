#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/math.hpp"
#include "driver-includes/timer_cuda.hpp"
#define BLOCK_SIZE 16
#define LEVELS_PER_THREAD 1
using namespace gridtools::dawn;


namespace dawn_generated{
namespace cuda_ico{
template<int E_C_SIZE, int E_V_SIZE>__global__ void reductions_stencil35_ms34_s32_kernel(int NumCells, int NumEdges, int NumVertices, int kSize, const int *ecTable, const int *evTable, ::dawn::float_type * __restrict__ lhs_field, const ::dawn::float_type * __restrict__ rhs_field, const ::dawn::float_type * __restrict__ cell_field, const ::dawn::float_type * __restrict__ node_field) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int kidx = blockIdx.y * blockDim.y + threadIdx.y;
  int klo = kidx * LEVELS_PER_THREAD;
  int khi = (kidx + 1) * LEVELS_PER_THREAD;
if (pidx >= NumCells) {
    return;
}for(int kIter = klo; kIter < khi; kIter++) {
  if (kIter >= kSize) {
      return;
  }::dawn::float_type lhs_18 = (::dawn::float_type) 0.000000;
for (int nbhIter = 0; nbhIter < E_C_SIZE; nbhIter++){
int nbhIdx = ecTable[pidx * E_C_SIZE + nbhIter];
if (nbhIdx == DEVICE_MISSING_VALUE) { continue; }lhs_18 +=cell_field[kIter * NumCells + pidx];}
::dawn::float_type lhs_11 = (::dawn::float_type) 0.000000;
for (int nbhIter = 0; nbhIter < E_V_SIZE; nbhIter++){
int nbhIdx = evTable[pidx * E_V_SIZE + nbhIter];
if (nbhIdx == DEVICE_MISSING_VALUE) { continue; }lhs_11 +=node_field[kIter * NumVertices + pidx];}
lhs_field[kIter * NumEdges + pidx] = ((rhs_field[kIter * NumEdges + pidx] +  lhs_18 ) +  lhs_11 );
}}
template<typename LibTag, int E_C_SIZE , int E_V_SIZE >
class reductions {
public:

  struct sbase : public timer_cuda {

    sbase(std::string name) : timer_cuda(name){}

    double get_time() {
      return total_time();
    }
  };

  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int* ecTable;
    int* evTable;

    GpuTriMesh(const dawn::mesh_t<LibTag>& mesh) {
      NumVertices = mesh.nodes().size();
      NumCells = mesh.cells().size();
      NumEdges = mesh.edges().size();
      gpuErrchk(cudaMalloc((void**)&ecTable, sizeof(int) * mesh.edges().size()* E_C_SIZE));
      dawn::generateNbhTable<LibTag>(mesh, {dawn::LocationType::Edges, dawn::LocationType::Cells}, mesh.edges().size(), E_C_SIZE, ecTable);
      gpuErrchk(cudaMalloc((void**)&evTable, sizeof(int) * mesh.edges().size()* E_V_SIZE));
      dawn::generateNbhTable<LibTag>(mesh, {dawn::LocationType::Edges, dawn::LocationType::Vertices}, mesh.edges().size(), E_V_SIZE, evTable);
    }
  };

  struct stencil_35 : public sbase {
  private:
    ::dawn::float_type* lhs_field_;
    ::dawn::float_type* rhs_field_;
    ::dawn::float_type* cell_field_;
    ::dawn::float_type* node_field_;
    int kSize_ = 0;
    GpuTriMesh mesh_;
  public:

    stencil_35(const dawn::mesh_t<LibTag>& mesh, int kSize, dawn::edge_field_t<LibTag, ::dawn::float_type>& lhs_field, dawn::edge_field_t<LibTag, ::dawn::float_type>& rhs_field, dawn::cell_field_t<LibTag, ::dawn::float_type>& cell_field, dawn::vertex_field_t<LibTag, ::dawn::float_type>& node_field) : sbase("stencil_35"), mesh_(mesh), kSize_(kSize){
      dawn::initField(lhs_field, &lhs_field_, mesh.edges().size(), kSize);
      dawn::initField(rhs_field, &rhs_field_, mesh.edges().size(), kSize);
      dawn::initField(cell_field, &cell_field_, mesh.cells().size(), kSize);
      dawn::initField(node_field, &node_field_, mesh.nodes().size(), kSize);
    }

    void run() {
      int dK = (kSize_ + LEVELS_PER_THREAD - 1) / LEVELS_PER_THREAD;
      dim3 dGC((mesh_.NumCells + BLOCK_SIZE - 1) / BLOCK_SIZE, (dK + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
      dim3 dB(BLOCK_SIZE, BLOCK_SIZE, 1);
      sbase::start();
      reductions_stencil35_ms34_s32_kernel<E_C_SIZE, E_V_SIZE><<<dGC,dB>>>(mesh_.NumCells, mesh_.NumEdges, mesh_.NumVertices, kSize_, mesh_.ecTable, mesh_.evTable, lhs_field_, rhs_field_, cell_field_, node_field_);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      sbase::pause();
    }

    void CopyResultToHost(dawn::edge_field_t<LibTag, ::dawn::float_type>& lhs_field) {
     {
        ::dawn::float_type* host_buf = new ::dawn::float_type[lhs_field.numElements()];
        gpuErrchk(cudaMemcpy((::dawn::float_type*) host_buf, lhs_field_, lhs_field.numElements()*sizeof(::dawn::float_type), cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, lhs_field.data(), kSize_, mesh_.NumEdges);
        delete[] host_buf;
    }    }
  };
};
} // namespace cuda_ico
} // namespace dawn_generated
