//---- Preprocessor defines ----
#define DAWN_GENERATED 1
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

//---- Includes ----
#include "driver-includes/gridtools_includes.hpp"
using namespace gridtools::dawn;

//---- Globals ----

//---- Stencils ----
namespace dawn_generated{
namespace cuda{
__global__ void __launch_bounds__(256)  hori_diff_stencil_stencil41_ms87_kernel(const int isize, const int jsize, const int ksize, const int stride_111_1, const int stride_111_2, ::dawn::float_type * const in, ::dawn::float_type * const out, ::dawn::float_type * const coeff) {

  // Start kernel
  __shared__ ::dawn::float_type lap_ijcache[34*6];
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
  int iblock = -1 - 1;
  int jblock = -1 - 1;
if(threadIdx.y < +6) {
    iblock = threadIdx.x;
    jblock = (int)threadIdx.y + -1;
}else if(threadIdx.y < +7) {
    iblock = -1 + (int)threadIdx.x % 1;
    jblock = (int)threadIdx.x / 1+-1;
}else if(threadIdx.y < 8) {
    iblock = threadIdx.x % 1 + 32;
    jblock = (int)threadIdx.x / 1+-1;
}
  // initialized iterators
  int idx111 = (blockIdx.x*32+iblock)*1+(blockIdx.y*4+jblock)*stride_111_1;
  int ijcacheindex= iblock + 1 + (jblock + 1)*34;

  // jump iterators to match the intersection of beginning of next interval and the parallel execution block 
  idx111 += max(0, blockIdx.z * 4) * stride_111_2;
  int kleg_lower_bound = max(0,blockIdx.z*4);
  int kleg_upper_bound = min( ksize - 1 + 0,(blockIdx.z+1)*4-1);;
for(int k = kleg_lower_bound+0; k <= kleg_upper_bound+0; ++k) {
  if(iblock >= -1 && iblock <= block_size_i -1 + 1 && jblock >= -1 && jblock <= block_size_j -1 + 1) {
lap_ijcache[ijcacheindex] = (((::dawn::float_type) -4.0 * __ldg(&(in[idx111]))) + (__ldg(&(coeff[idx111])) * (__ldg(&(in[idx111+1*1])) + (__ldg(&(in[idx111+1*-1])) + (__ldg(&(in[idx111+stride_111_1*1])) + __ldg(&(in[idx111+stride_111_1*-1])))))));
  }    __syncthreads();
  if(iblock >= 0 && iblock <= block_size_i -1 + 0 && jblock >= 0 && jblock <= block_size_j -1 + 0) {
out[idx111] = (((::dawn::float_type) -4.0 * lap_ijcache[ijcacheindex]) + (__ldg(&(coeff[idx111])) * (lap_ijcache[ijcacheindex+1] + (lap_ijcache[ijcacheindex+-1] + (lap_ijcache[ijcacheindex+1*34] + lap_ijcache[ijcacheindex+-1*34])))));
  }    __syncthreads();

    // Slide kcaches

    // increment iterators
    idx111+=stride_111_2;
}}

class hori_diff_stencil {
public:

  struct sbase : public timer_cuda {

    sbase(std::string name) : timer_cuda(name){}

    double get_time() {
      return total_time();
    }
  };

  struct stencil_41 : public sbase {

    // Members

    // Temporary storage typedefs
    using tmp_halo_t = gridtools::halo< 1,1, 0, 0, 0>;
    using tmp_meta_data_t = storage_traits_t::storage_info_t< 0, 5, tmp_halo_t >;
    using tmp_storage_t = storage_traits_t::data_store_t< ::dawn::float_type, tmp_meta_data_t>;
    const gridtools::dawn::domain m_dom;

    // temporary storage declarations
    tmp_meta_data_t m_tmp_meta_data;
    tmp_storage_t m_lap;
  public:

    stencil_41(const gridtools::dawn::domain& dom_, int rank, int xcols, int ycols) : sbase("stencil_41"), m_dom(dom_), m_tmp_meta_data(32+2, 4+2, (dom_.isize()+ 32 - 1) / 32, (dom_.jsize()+ 4 - 1) / 4, dom_.ksize() + 2 * 0), m_lap(m_tmp_meta_data){}

    void run(storage_ijk_t in_ds, storage_ijk_t out_ds, storage_ijk_t coeff_ds) {

      // starting timers
      start();
      {;
      gridtools::data_view<storage_ijk_t> in= gridtools::make_device_view(in_ds);
      gridtools::data_view<storage_ijk_t> out= gridtools::make_device_view(out_ds);
      gridtools::data_view<storage_ijk_t> coeff= gridtools::make_device_view(coeff_ds);
      const unsigned int nx = m_dom.isize() - m_dom.iminus() - m_dom.iplus();
      const unsigned int ny = m_dom.jsize() - m_dom.jminus() - m_dom.jplus();
      const unsigned int nz = m_dom.ksize() - m_dom.kminus() - m_dom.kplus();
      dim3 threads(32,4+4,1);
      const unsigned int nbx = (nx + 32 - 1) / 32;
      const unsigned int nby = (ny + 4 - 1) / 4;
      const unsigned int nbz = (m_dom.ksize()+4-1) / 4;
      dim3 blocks(nbx, nby, nbz);
      hori_diff_stencil_stencil41_ms87_kernel<<<blocks, threads>>>(nx,ny,nz,in_ds.strides()[1],in_ds.strides()[2],(in.data()+in_ds.get_storage_info_ptr()->index(in.begin<0>(), in.begin<1>(),0 )),(out.data()+out_ds.get_storage_info_ptr()->index(out.begin<0>(), out.begin<1>(),0 )),(coeff.data()+coeff_ds.get_storage_info_ptr()->index(coeff.begin<0>(), coeff.begin<1>(),0 )));
      };

      // stopping timers
      pause();
    }
  };
  static constexpr const char* s_name = "hori_diff_stencil";
  stencil_41 m_stencil_41;
public:

  hori_diff_stencil(const hori_diff_stencil&) = delete;

  // Members

  // Stencil-Data

  hori_diff_stencil(const gridtools::dawn::domain& dom, int rank = 1, int xcols = 1, int ycols = 1) : m_stencil_41(dom, rank, xcols, ycols){}

  template<typename S>
  void sync_storages(S field) {
    field.sync();
  }

  template<typename S0, typename ... S>
  void sync_storages(S0 f0, S... fields) {
    f0.sync();
    sync_storages(fields...);
  }

  void run(storage_ijk_t in, storage_ijk_t out, storage_ijk_t coeff) {
    sync_storages(in,out,coeff);
    m_stencil_41.run(in,out,coeff);
;
    sync_storages(in,out,coeff);
  }

  std::string get_name()  const {
    return std::string(s_name);
  }

  void reset_meters() {
m_stencil_41.reset();  }

  double get_total_time() {
    double res = 0;
    res +=m_stencil_41.get_time();
    return res;
  }
};
} // namespace cuda
} // namespace dawn_generated
