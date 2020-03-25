#define DAWN_GENERATED 1
#undef DAWN_BACKEND_T
#define DAWN_BACKEND_T CUDA
#ifndef BOOST_RESULT_OF_USE_TR1
 #define BOOST_RESULT_OF_USE_TR1 1
#endif
#ifndef BOOST_NO_CXX11_DECLTYPE
 #define BOOST_NO_CXX11_DECLTYPE 1
#endif
#ifndef GRIDTOOLS_DAWN_HALO_EXTENT
 #define GRIDTOOLS_DAWN_HALO_EXTENT 3
#endif
#ifndef BOOST_PP_VARIADICS
 #define BOOST_PP_VARIADICS 1
#endif
#ifndef BOOST_FUSION_DONT_USE_PREPROCESSED_FILES
 #define BOOST_FUSION_DONT_USE_PREPROCESSED_FILES 1
#endif
#ifndef BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
 #define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS 1
#endif
#ifndef GT_VECTOR_LIMIT_SIZE
 #define GT_VECTOR_LIMIT_SIZE 30
#endif
#ifndef BOOST_FUSION_INVOKE_MAX_ARITY
 #define BOOST_FUSION_INVOKE_MAX_ARITY GT_VECTOR_LIMIT_SIZE
#endif
#ifndef FUSION_MAX_VECTOR_SIZE
 #define FUSION_MAX_VECTOR_SIZE GT_VECTOR_LIMIT_SIZE
#endif
#ifndef FUSION_MAX_MAP_SIZE
 #define FUSION_MAX_MAP_SIZE GT_VECTOR_LIMIT_SIZE
#endif
#ifndef BOOST_MPL_LIMIT_VECTOR_SIZE
 #define BOOST_MPL_LIMIT_VECTOR_SIZE GT_VECTOR_LIMIT_SIZE
#endif
#include <driver-includes/gridtools_includes.hpp>
using namespace gridtools::dawn;


namespace dawn_generated{
namespace cuda{
__constant__ int stage120GlobalIIndices_[2];
__constant__ int stage124GlobalIIndices_[2];
__constant__ int stage128GlobalJIndices_[2];
__constant__ int stage132GlobalJIndices_[2];
__constant__ int stage136GlobalIIndices_[2];
__constant__ int stage136GlobalJIndices_[2];
__constant__ int stage140GlobalIIndices_[2];
__constant__ int stage140GlobalJIndices_[2];
__constant__ int stage144GlobalIIndices_[2];
__constant__ int stage144GlobalJIndices_[2];
__constant__ int stage148GlobalIIndices_[2];
__constant__ int stage148GlobalJIndices_[2];
__constant__ unsigned globalOffsets_[2];
__device__ bool checkOffset(unsigned int min, unsigned int max, unsigned int val) {
  return (min <= val && val < max);
}
__global__ void __launch_bounds__(128)  global_index_stencil_stencil59_ms151_kernel(const int isize, const int jsize, const int ksize, const int stride_111_1, const int stride_111_2, ::dawn::float_type * const in, ::dawn::float_type * const out) {

  // Start kernel
  const unsigned int nx = isize;
  const unsigned int ny = jsize;
  const int block_size_i = (blockIdx.x + 1) * 32 < nx ? 32 : nx - blockIdx.x * 32;
  const int block_size_j = (blockIdx.y + 1) * 4 < ny ? 4 : ny - blockIdx.y * 4;

  // computing the global position in the physical domain

  // In a typical cuda block we have the following regions

  // aa bbbbbbbb cc

  // aa bbbbbbbb cc

  // hh dddddddd ii

  // hh dddddddd ii

  // hh dddddddd ii

  // hh dddddddd ii

  // ee ffffffff gg

  // ee ffffffff gg

  // Regions b,d,f have warp (or multiple of warp size)

  // Size of regions a, c, h, i, e, g are determined by max_extent_t

  // Regions b,d,f are easily executed by dedicated warps (one warp for each line)

  // Regions (a,h,e) and (c,i,g) are executed by two specialized warp
  int iblock = 0 - 1;
  int jblock = 0 - 1;
if(threadIdx.y < +4) {
    iblock = threadIdx.x;
    jblock = (int)threadIdx.y + 0;
}
  // initialized iterators
  int idx111 = (blockIdx.x*32+iblock)*1+(blockIdx.y*4+jblock)*stride_111_1;

  // jump iterators to match the intersection of beginning of next interval and the parallel execution block 
  idx111 += max(0, blockIdx.z * 4) * stride_111_2;
  int kleg_lower_bound = max(0,blockIdx.z*4);
  int kleg_upper_bound = min( ksize - 1 + 0,(blockIdx.z+1)*4-1);;
for(int k = kleg_lower_bound+0; k <= kleg_upper_bound+0; ++k) {
  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
out[idx111] = __ldg(&(in[idx111]));
  }  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0 && checkOffset(stage120GlobalIIndices_[0], stage120GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage124GlobalIIndices_[0], stage124GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage128GlobalJIndices_[0], stage128GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage132GlobalJIndices_[0], stage132GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage136GlobalIIndices_[0], stage136GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage136GlobalJIndices_[0], stage136GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage140GlobalIIndices_[0], stage140GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage140GlobalJIndices_[0], stage140GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage144GlobalIIndices_[0], stage144GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage144GlobalJIndices_[0], stage144GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage148GlobalIIndices_[0], stage148GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage148GlobalJIndices_[0], stage148GlobalJIndices_[1], globalOffsets_[1] + jblock)) {
out[idx111] = (::dawn::float_type) 4;
  }  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0 && checkOffset(stage120GlobalIIndices_[0], stage120GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage124GlobalIIndices_[0], stage124GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage128GlobalJIndices_[0], stage128GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage132GlobalJIndices_[0], stage132GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage136GlobalIIndices_[0], stage136GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage136GlobalJIndices_[0], stage136GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage140GlobalIIndices_[0], stage140GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage140GlobalJIndices_[0], stage140GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage144GlobalIIndices_[0], stage144GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage144GlobalJIndices_[0], stage144GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage148GlobalIIndices_[0], stage148GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage148GlobalJIndices_[0], stage148GlobalJIndices_[1], globalOffsets_[1] + jblock)) {
out[idx111] = (::dawn::float_type) 8;
  }  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0 && checkOffset(stage120GlobalIIndices_[0], stage120GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage124GlobalIIndices_[0], stage124GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage128GlobalJIndices_[0], stage128GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage132GlobalJIndices_[0], stage132GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage136GlobalIIndices_[0], stage136GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage136GlobalJIndices_[0], stage136GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage140GlobalIIndices_[0], stage140GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage140GlobalJIndices_[0], stage140GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage144GlobalIIndices_[0], stage144GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage144GlobalJIndices_[0], stage144GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage148GlobalIIndices_[0], stage148GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage148GlobalJIndices_[0], stage148GlobalJIndices_[1], globalOffsets_[1] + jblock)) {
out[idx111] = (::dawn::float_type) 6;
  }  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0 && checkOffset(stage120GlobalIIndices_[0], stage120GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage124GlobalIIndices_[0], stage124GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage128GlobalJIndices_[0], stage128GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage132GlobalJIndices_[0], stage132GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage136GlobalIIndices_[0], stage136GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage136GlobalJIndices_[0], stage136GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage140GlobalIIndices_[0], stage140GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage140GlobalJIndices_[0], stage140GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage144GlobalIIndices_[0], stage144GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage144GlobalJIndices_[0], stage144GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage148GlobalIIndices_[0], stage148GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage148GlobalJIndices_[0], stage148GlobalJIndices_[1], globalOffsets_[1] + jblock)) {
out[idx111] = (::dawn::float_type) 2;
  }  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0 && checkOffset(stage120GlobalIIndices_[0], stage120GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage124GlobalIIndices_[0], stage124GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage128GlobalJIndices_[0], stage128GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage132GlobalJIndices_[0], stage132GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage136GlobalIIndices_[0], stage136GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage136GlobalJIndices_[0], stage136GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage140GlobalIIndices_[0], stage140GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage140GlobalJIndices_[0], stage140GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage144GlobalIIndices_[0], stage144GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage144GlobalJIndices_[0], stage144GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage148GlobalIIndices_[0], stage148GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage148GlobalJIndices_[0], stage148GlobalJIndices_[1], globalOffsets_[1] + jblock)) {
out[idx111] = (::dawn::float_type) 1;
  }  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0 && checkOffset(stage120GlobalIIndices_[0], stage120GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage124GlobalIIndices_[0], stage124GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage128GlobalJIndices_[0], stage128GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage132GlobalJIndices_[0], stage132GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage136GlobalIIndices_[0], stage136GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage136GlobalJIndices_[0], stage136GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage140GlobalIIndices_[0], stage140GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage140GlobalJIndices_[0], stage140GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage144GlobalIIndices_[0], stage144GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage144GlobalJIndices_[0], stage144GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage148GlobalIIndices_[0], stage148GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage148GlobalJIndices_[0], stage148GlobalJIndices_[1], globalOffsets_[1] + jblock)) {
out[idx111] = (::dawn::float_type) 3;
  }  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0 && checkOffset(stage120GlobalIIndices_[0], stage120GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage124GlobalIIndices_[0], stage124GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage128GlobalJIndices_[0], stage128GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage132GlobalJIndices_[0], stage132GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage136GlobalIIndices_[0], stage136GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage136GlobalJIndices_[0], stage136GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage140GlobalIIndices_[0], stage140GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage140GlobalJIndices_[0], stage140GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage144GlobalIIndices_[0], stage144GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage144GlobalJIndices_[0], stage144GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage148GlobalIIndices_[0], stage148GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage148GlobalJIndices_[0], stage148GlobalJIndices_[1], globalOffsets_[1] + jblock)) {
out[idx111] = (::dawn::float_type) 7;
  }  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0 && checkOffset(stage120GlobalIIndices_[0], stage120GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage124GlobalIIndices_[0], stage124GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage128GlobalJIndices_[0], stage128GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage132GlobalJIndices_[0], stage132GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage136GlobalIIndices_[0], stage136GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage136GlobalJIndices_[0], stage136GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage140GlobalIIndices_[0], stage140GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage140GlobalJIndices_[0], stage140GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage144GlobalIIndices_[0], stage144GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage144GlobalJIndices_[0], stage144GlobalJIndices_[1], globalOffsets_[1] + jblock) && checkOffset(stage148GlobalIIndices_[0], stage148GlobalIIndices_[1], globalOffsets_[0] + iblock) && checkOffset(stage148GlobalJIndices_[0], stage148GlobalJIndices_[1], globalOffsets_[1] + jblock)) {
out[idx111] = (::dawn::float_type) 5;
  }
    // Slide kcaches

    // increment iterators
    idx111+=stride_111_2;
}}

class global_index_stencil {
public:

  struct sbase : public timer_cuda {

    sbase(std::string name) : timer_cuda(name){}

    double get_time() {
      return total_time();
    }
  };

  struct stencil_59 : public sbase {

    // Members
    std::array<int, 2> stage120GlobalIIndices;
    std::array<int, 2> stage124GlobalIIndices;
    std::array<int, 2> stage128GlobalJIndices;
    std::array<int, 2> stage132GlobalJIndices;
    std::array<int, 2> stage136GlobalIIndices;
    std::array<int, 2> stage136GlobalJIndices;
    std::array<int, 2> stage140GlobalIIndices;
    std::array<int, 2> stage140GlobalJIndices;
    std::array<int, 2> stage144GlobalIIndices;
    std::array<int, 2> stage144GlobalJIndices;
    std::array<int, 2> stage148GlobalIIndices;
    std::array<int, 2> stage148GlobalJIndices;
    std::array<unsigned int, 2> globalOffsets;

    static std::array<unsigned int, 2> computeGlobalOffsets(int rank, const gridtools::dawn::domain& dom, int xcols, int ycols) {
      unsigned int rankOnDefaultFace = rank % (xcols * ycols);
      unsigned int row = rankOnDefaultFace / xcols;
      unsigned int col = rankOnDefaultFace % ycols;
      return {col * (dom.isize() - dom.iplus()), row * (dom.jsize() - dom.jplus())};
    }

    // Temporary storage typedefs
    using tmp_halo_t = gridtools::halo< 0,0, 0, 0, 0>;
    using tmp_meta_data_t = storage_traits_t::storage_info_t< 0, 5, tmp_halo_t >;
    using tmp_storage_t = storage_traits_t::data_store_t< ::dawn::float_type, tmp_meta_data_t>;
    const gridtools::dawn::domain m_dom;
  public:

    stencil_59(const gridtools::dawn::domain& dom_, int rank, int xcols, int ycols) : sbase("stencil_59"), m_dom(dom_), stage120GlobalIIndices({dom_.isize() - dom_.iplus()  + -1 , dom_.isize() - dom_.iplus()  + 0}), stage124GlobalIIndices({dom_.iminus() + 0 , dom_.iminus() + 1}), stage128GlobalJIndices({dom_.jsize() - dom_.jplus()  + -1 , dom_.jsize() - dom_.jplus()  + 0}), stage132GlobalJIndices({dom_.jminus() + 0 , dom_.jminus() + 1}), stage136GlobalIIndices({dom_.iminus() + 0 , dom_.iminus() + 1}), stage136GlobalJIndices({dom_.jminus() + 0 , dom_.jminus() + 1}), stage140GlobalIIndices({dom_.isize() - dom_.iplus()  + -1 , dom_.isize() - dom_.iplus()  + 0}), stage140GlobalJIndices({dom_.jminus() + 0 , dom_.jminus() + 1}), stage144GlobalIIndices({dom_.iminus() + 0 , dom_.iminus() + 1}), stage144GlobalJIndices({dom_.jsize() - dom_.jplus()  + -1 , dom_.jsize() - dom_.jplus()  + 0}), stage148GlobalIIndices({dom_.isize() - dom_.iplus()  + -1 , dom_.isize() - dom_.iplus()  + 0}), stage148GlobalJIndices({dom_.jsize() - dom_.jplus()  + -1 , dom_.jsize() - dom_.jplus()  + 0}), globalOffsets({computeGlobalOffsets(rank, m_dom, xcols, ycols)}){}
    static constexpr dawn::driver::cartesian_extent in_extent = {0,0, 0,0, 0,0};
    static constexpr dawn::driver::cartesian_extent out_extent = {0,0, 0,0, 0,0};

    void run(storage_ijk_t in_ds, storage_ijk_t out_ds) {

      // starting timers
      start();
      {;
      gridtools::data_view<storage_ijk_t> in= gridtools::make_device_view(in_ds);
      gridtools::data_view<storage_ijk_t> out= gridtools::make_device_view(out_ds);
      const unsigned int nx = m_dom.isize() - m_dom.iminus() - m_dom.iplus();
      const unsigned int ny = m_dom.jsize() - m_dom.jminus() - m_dom.jplus();
      const unsigned int nz = m_dom.ksize() - m_dom.kminus() - m_dom.kplus();
      dim3 threads(32,4+0,1);
      const unsigned int nbx = (nx + 32 - 1) / 32;
      const unsigned int nby = (ny + 4 - 1) / 4;
      const unsigned int nbz = (m_dom.ksize()+4-1) / 4;
      cudaMemcpyToSymbol(stage120GlobalIIndices_, stage120GlobalIIndices.data(), sizeof(int) * stage120GlobalIIndices.size());
      cudaMemcpyToSymbol(stage124GlobalIIndices_, stage124GlobalIIndices.data(), sizeof(int) * stage124GlobalIIndices.size());
      cudaMemcpyToSymbol(stage128GlobalJIndices_, stage128GlobalJIndices.data(), sizeof(int) * stage128GlobalJIndices.size());
      cudaMemcpyToSymbol(stage132GlobalJIndices_, stage132GlobalJIndices.data(), sizeof(int) * stage132GlobalJIndices.size());
      cudaMemcpyToSymbol(stage136GlobalIIndices_, stage136GlobalIIndices.data(), sizeof(int) * stage136GlobalIIndices.size());
      cudaMemcpyToSymbol(stage136GlobalJIndices_, stage136GlobalJIndices.data(), sizeof(int) * stage136GlobalJIndices.size());
      cudaMemcpyToSymbol(stage140GlobalIIndices_, stage140GlobalIIndices.data(), sizeof(int) * stage140GlobalIIndices.size());
      cudaMemcpyToSymbol(stage140GlobalJIndices_, stage140GlobalJIndices.data(), sizeof(int) * stage140GlobalJIndices.size());
      cudaMemcpyToSymbol(stage144GlobalIIndices_, stage144GlobalIIndices.data(), sizeof(int) * stage144GlobalIIndices.size());
      cudaMemcpyToSymbol(stage144GlobalJIndices_, stage144GlobalJIndices.data(), sizeof(int) * stage144GlobalJIndices.size());
      cudaMemcpyToSymbol(stage148GlobalIIndices_, stage148GlobalIIndices.data(), sizeof(int) * stage148GlobalIIndices.size());
      cudaMemcpyToSymbol(stage148GlobalJIndices_, stage148GlobalJIndices.data(), sizeof(int) * stage148GlobalJIndices.size());
      cudaMemcpyToSymbol(globalOffsets_, globalOffsets.data(), sizeof(unsigned) * globalOffsets.size());
      dim3 blocks(nbx, nby, nbz);
      global_index_stencil_stencil59_ms151_kernel<<<blocks, threads>>>(nx,ny,nz,in_ds.strides()[1],in_ds.strides()[2],(in.data()+in_ds.get_storage_info_ptr()->index(in.begin<0>(), in.begin<1>(),0 )),(out.data()+out_ds.get_storage_info_ptr()->index(out.begin<0>(), out.begin<1>(),0 )));
      };

      // stopping timers
      pause();
    }
  };
  static constexpr const char* s_name = "global_index_stencil";
  stencil_59 m_stencil_59;
public:

  global_index_stencil(const global_index_stencil&) = delete;

  // Members

  // Stencil-Data

  global_index_stencil(const gridtools::dawn::domain& dom, int rank = 1, int xcols = 1, int ycols = 1) : m_stencil_59(dom, rank, xcols, ycols){}

  template<typename S>
  void sync_storages(S field) {
    field.sync();
  }

  template<typename S0, typename ... S>
  void sync_storages(S0 f0, S... fields) {
    f0.sync();
    sync_storages(fields...);
  }

  void run(storage_ijk_t in, storage_ijk_t out) {
    sync_storages(in,out);
    m_stencil_59.run(in,out);
;
    sync_storages(in,out);
  }

  std::string get_name()  const {
    return std::string(s_name);
  }

  void reset_meters() {
m_stencil_59.reset();  }

  double get_total_time() {
    double res = 0;
    res +=m_stencil_59.get_time();
    return res;
  }
};
} // namespace cuda
} // namespace dawn_generated

