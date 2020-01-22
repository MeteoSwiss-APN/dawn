#define DAWN_GENERATED 1
#define DAWN_BACKEND_T CUDA
#ifndef BOOST_RESULT_OF_USE_TR1
 #define BOOST_RESULT_OF_USE_TR1 1
#endif
#ifndef BOOST_NO_CXX11_DECLTYPE
 #define BOOST_NO_CXX11_DECLTYPE 1
#endif
#ifndef GRIDTOOLS_DAWN_HALO_EXTENT
 #define GRIDTOOLS_DAWN_HALO_EXTENT 0
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
__device__ bool checkOffset(unsigned int min, unsigned int max, unsigned int val) {
  return (min <= val && val < max);
}
__global__ void __launch_bounds__(128)  generated_stencil28_ms27_kernel(const int isize, const int jsize, const int ksize, const int stride_111_1, const int stride_111_2, ::dawn::float_type * const in_field, ::dawn::float_type * const out_field, int* const stage14GlobalJIndices, unsigned* const globalOffsets) {

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
{
  out_field[idx111] = __ldg(&(in_field[idx111]));
}
  }  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0 && checkOffset(stage14GlobalJIndices[0], stage14GlobalJIndices[1], globalOffsets[1] + jblock)) {
{
  out_field[idx111] = (int) 10;
}
  }
    // Slide kcaches

    // increment iterators
    idx111+=stride_111_2;
}}

class generated {
public:

  struct sbase : public timer_cuda {

    sbase(std::string name) : timer_cuda(name){}

    double get_time() {
      return total_time();
    }
  };

  struct stencil_28 : public sbase {

    // Members
    std::array<int, 2> stage14GlobalJIndices;
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

    stencil_28(const gridtools::dawn::domain& dom_, int rank, int xcols, int ycols) : sbase("stencil_28"), m_dom(dom_), stage14GlobalJIndices({dom_.jminus() + 0 , dom_.jminus() + 2}), globalOffsets({computeGlobalOffsets(rank, m_dom, xcols, ycols)}){}

    void run(storage_ijk_t in_field_ds, storage_ijk_t out_field_ds) {

      // starting timers
      start();
      {;
      gridtools::data_view<storage_ijk_t> in_field= gridtools::make_device_view(in_field_ds);
      gridtools::data_view<storage_ijk_t> out_field= gridtools::make_device_view(out_field_ds);
      const unsigned int nx = m_dom.isize() - m_dom.iminus() - m_dom.iplus();
      const unsigned int ny = m_dom.jsize() - m_dom.jminus() - m_dom.jplus();
      const unsigned int nz = m_dom.ksize() - m_dom.kminus() - m_dom.kplus();
      dim3 threads(32,4+0,1);
      const unsigned int nbx = (nx + 32 - 1) / 32;
      const unsigned int nby = (ny + 4 - 1) / 4;
      const unsigned int nbz = (m_dom.ksize()+4-1) / 4;
      dim3 blocks(nbx, nby, nbz);
      generated_stencil28_ms27_kernel<<<blocks, threads>>>(nx,ny,nz,in_field_ds.strides()[1],in_field_ds.strides()[2],(in_field.data()+in_field_ds.get_storage_info_ptr()->index(in_field.begin<0>(), in_field.begin<1>(),0 )),(out_field.data()+out_field_ds.get_storage_info_ptr()->index(out_field.begin<0>(), out_field.begin<1>(),0 )), stage14GlobalJIndices.data(), globalOffsets.data());
      };

      // stopping timers
      pause();
    }
  };
  static constexpr const char* s_name = "generated";
  stencil_28 m_stencil_28;
public:

  generated(const generated&) = delete;

  // Members

  // Stencil-Data

  generated(const gridtools::dawn::domain& dom, int rank = 1, int xcols = 1, int ycols = 1) : m_stencil_28(dom, rank, xcols, ycols){}

  template<typename S>
  void sync_storages(S field) {
    field.sync();
  }

  template<typename S0, typename ... S>
  void sync_storages(S0 f0, S... fields) {
    f0.sync();
    sync_storages(fields...);
  }

  void run(storage_ijk_t in_field, storage_ijk_t out_field) {
    sync_storages(in_field,out_field);
    m_stencil_28.run(in_field,out_field);
;
    sync_storages(in_field,out_field);
  }

  std::string get_name()  const {
    return std::string(s_name);
  }

  void reset_meters() {
m_stencil_28.reset();  }

  double get_total_time() {
    double res = 0;
    res +=m_stencil_28.get_time();
    return res;
  }
};
} // namespace cuda
} // namespace dawn_generated
