#define DAWN_GENERATED 1
#undef DAWN_BACKEND_T
#define DAWN_BACKEND_T CXXOPT
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
#include <omp.h>


namespace dawn_generated{
namespace cxxopt{

class generated {
private:

  struct stencil_47 {

    // Members

    // Temporary storages
    using tmp_halo_t = gridtools::halo< GRIDTOOLS_DAWN_HALO_EXTENT, GRIDTOOLS_DAWN_HALO_EXTENT, 0>;
    using tmp_meta_data_t = storage_traits_t::storage_info_t< 0, 3, tmp_halo_t >;
    using tmp_storage_t = storage_traits_t::data_store_t< ::dawn::float_type, tmp_meta_data_t>;
    const gridtools::dawn::domain m_dom;

    // Input/Output storages
  public:

    stencil_47(const gridtools::dawn::domain& dom_, int rank, int xcols, int ycols) : m_dom(dom_){}
    static constexpr dawn::driver::cartesian_extent in_extent = {-1,1, -1,1, 0,0};
    static constexpr dawn::driver::cartesian_extent out_extent = {0,0, 0,0, 0,0};

    void run(storage_ijk_t& in_, storage_ijk_t& out_) {
      int iMin = m_dom.iminus();
      int iMax = m_dom.isize() - m_dom.iplus() - 1;
      int jMin = m_dom.jminus();
      int jMax = m_dom.jsize() - m_dom.jplus() - 1;
      int kMin = m_dom.kminus();
      int kMax = m_dom.ksize() - m_dom.kplus() - 1;
      in_.sync();
      out_.sync();
{      gridtools::data_view<storage_ijk_t> in= gridtools::make_host_view(in_);
      std::array<int,3> in_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> out= gridtools::make_host_view(out_);
      std::array<int,3> out_offsets{0,0,0};
    
#pragma omp parallel for
for(int k = kMin + 0+0; k <= kMax + 0+0; ++k) {
      for(int i = iMin+0; i  <=  iMax+0; ++i) {
        #pragma omp simd
for(int j = jMin+0; j  <=  jMax+0; ++j) {
::dawn::float_type dx;
{
  out(i+0, j+0, k+0) = (((int) -4 * (in(i+0, j+0, k+0) + (in(i+1, j+0, k+0) + (in(i+-1, j+0, k+0) + (in(i+0, j+-1, k+0) + in(i+0, j+1, k+0)))))) / (dx * dx));
}
        }      }    }}      in_.sync();
      out_.sync();
    }
  };
  static constexpr const char* s_name = "generated";
  stencil_47 m_stencil_47;
public:

  generated(const generated&) = delete;

  generated(const gridtools::dawn::domain& dom, int rank = 1, int xcols = 1, int ycols = 1) : m_stencil_47(dom, rank, xcols, ycols){
    assert(dom.isize() >= dom.iminus() + dom.iplus());
    assert(dom.jsize() >= dom.jminus() + dom.jplus());
    assert(dom.ksize() >= dom.kminus() + dom.kplus());
    assert(dom.ksize() >= 1);
  }

  void run(storage_ijk_t in, storage_ijk_t out) {
    m_stencil_47.run(in,out);
  }
};
} // namespace cxxopt
} // namespace dawn_generated
