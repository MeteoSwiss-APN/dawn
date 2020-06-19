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

struct globals {
  int var1;
  bool var2;

  globals() : var1(1){
  }
};
} // namespace cuda
} // namespace dawn_generated


namespace dawn_generated{
namespace cuda{
__global__ void __launch_bounds__(128)  conditional_stencil_stencil21_ms41_kernel(globals globals_, const int isize, const int jsize, const int ksize, const int stride_111_1, const int stride_111_2, ::dawn::float_type * const in, ::dawn::float_type * const out) {

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
if((globals_.var1 == (int) 1))
{
  out[idx111] = __ldg(&(in[idx111+1*1]));
}
else
{
  out[idx111] = __ldg(&(in[idx111+1*-1]));
}
if((globals_.var1 == (int) 1))
{
  out[idx111] = __ldg(&(in[idx111+stride_111_1*1]));
}
else
{
  out[idx111] = __ldg(&(in[idx111+stride_111_1*-1]));
}
  }
    // Slide kcaches

    // increment iterators
    idx111+=stride_111_2;
}}

class conditional_stencil {
public:

  struct sbase : public timer_cuda {

    sbase(std::string name) : timer_cuda(name){}

    double get_time() {
      return total_time();
    }
  };

  struct stencil_21 : public sbase {

    // Members

    // Temporary storage typedefs
    using tmp_halo_t = gridtools::halo< 0,0, 0, 0, 0>;
    using tmp_meta_data_t = storage_traits_t::storage_info_t< 0, 5, tmp_halo_t >;
    using tmp_storage_t = storage_traits_t::data_store_t< ::dawn::float_type, tmp_meta_data_t>;
    globals& m_globals;
    const gridtools::dawn::domain m_dom;
  public:

    stencil_21(const gridtools::dawn::domain& dom_, globals& globals_, int rank, int xcols, int ycols) : sbase("stencil_21"), m_dom(dom_), m_globals(globals_){}
    static constexpr dawn::driver::cartesian_extent in_extent = {-1,1, -1,1, 0,0};
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
      dim3 blocks(nbx, nby, nbz);
      conditional_stencil_stencil21_ms41_kernel<<<blocks, threads>>>(m_globals,nx,ny,nz,in_ds.strides()[1],in_ds.strides()[2],(in.data()+in_ds.get_storage_info_ptr()->index(in.begin<0>(), in.begin<1>(),0 )),(out.data()+out_ds.get_storage_info_ptr()->index(out.begin<0>(), out.begin<1>(),0 )));
      };

      // stopping timers
      pause();
    }
  };
  static constexpr const char* s_name = "conditional_stencil";
  stencil_21 m_stencil_21;
public:

  conditional_stencil(const conditional_stencil&) = delete;

  // Members

  // Stencil-Data
  globals m_globals;

  conditional_stencil(const gridtools::dawn::domain& dom, int rank = 1, int xcols = 1, int ycols = 1) : m_stencil_21(dom,m_globals, rank, xcols, ycols){}

  // Access-wrapper for globally defined variables

  int get_var1() {
    return m_globals.var1;
  }

  void set_var1(int var1) {
    m_globals.var1=var1;
  }

  bool get_var2() {
    return m_globals.var2;
  }

  void set_var2(bool var2) {
    m_globals.var2=var2;
  }

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
    m_stencil_21.run(in,out);
;
    sync_storages(in,out);
  }

  std::string get_name()  const {
    return std::string(s_name);
  }

  void reset_meters() {
m_stencil_21.reset();  }

  double get_total_time() {
    double res = 0;
    res +=m_stencil_21.get_time();
    return res;
  }
};
} // namespace cuda
} // namespace dawn_generated
