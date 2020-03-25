#define DAWN_GENERATED 1
#undef DAWN_BACKEND_T
#define DAWN_BACKEND_T CXXNAIVE
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
namespace cxxnaive{

class tridiagonal_solve_stencil {
private:

  struct stencil_49 {

    // Members

    // Temporary storages
    using tmp_halo_t = gridtools::halo< GRIDTOOLS_DAWN_HALO_EXTENT, GRIDTOOLS_DAWN_HALO_EXTENT, 0>;
    using tmp_meta_data_t = storage_traits_t::storage_info_t< 0, 3, tmp_halo_t >;
    using tmp_storage_t = storage_traits_t::data_store_t< ::dawn::float_type, tmp_meta_data_t>;
    const gridtools::dawn::domain m_dom;

    // Input/Output storages
  public:

    stencil_49(const gridtools::dawn::domain& dom_, int rank, int xcols, int ycols) : m_dom(dom_){}
    static constexpr dawn::driver::cartesian_extent a_extent = {0,0, 0,0, 0,0};
    static constexpr dawn::driver::cartesian_extent b_extent = {0,0, 0,0, 0,0};
    static constexpr dawn::driver::cartesian_extent c_extent = {0,0, 0,0, -1,0};
    static constexpr dawn::driver::cartesian_extent d_extent = {0,0, 0,0, -1,1};

    void run(storage_ijk_t& a_, storage_ijk_t& b_, storage_ijk_t& c_, storage_ijk_t& d_) {
      int iMin = m_dom.iminus();
      int iMax = m_dom.isize() - m_dom.iplus() - 1;
      int jMin = m_dom.jminus();
      int jMax = m_dom.jsize() - m_dom.jplus() - 1;
      int kMin = m_dom.kminus();
      int kMax = m_dom.ksize() - m_dom.kplus() - 1;
      a_.sync();
      b_.sync();
      c_.sync();
      d_.sync();
{      gridtools::data_view<storage_ijk_t> a= gridtools::make_host_view(a_);
      std::array<int,3> a_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> b= gridtools::make_host_view(b_);
      std::array<int,3> b_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> c= gridtools::make_host_view(c_);
      std::array<int,3> c_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> d= gridtools::make_host_view(d_);
      std::array<int,3> d_offsets{0,0,0};
    for(int k = kMin + 0+0; k <= kMin + 0+0; ++k) {
      for(int i = iMin+0; i  <=  iMax+0; ++i) {
        for(int j = jMin+0; j  <=  jMax+0; ++j) {
c(i+0, j+0, k+0) = (c(i+0, j+0, k+0) / b(i+0, j+0, k+0));
        }      }    }    for(int k = kMin + 1+0; k <= kMax + 0+0; ++k) {
      for(int i = iMin+0; i  <=  iMax+0; ++i) {
        for(int j = jMin+0; j  <=  jMax+0; ++j) {
c(i+0, j+0, k+0) = (c(i+0, j+0, k+0) / b(i+0, j+0, k+0));
        }      }      for(int i = iMin+0; i  <=  iMax+0; ++i) {
        for(int j = jMin+0; j  <=  jMax+0; ++j) {
int __local_m_100 = ((::dawn::float_type) 1.0 / (b(i+0, j+0, k+0) - (a(i+0, j+0, k+0) * c(i+0, j+0, k+-1))));
c(i+0, j+0, k+0) = (c(i+0, j+0, k+0) * __local_m_100);
d(i+0, j+0, k+0) = ((d(i+0, j+0, k+0) - (a(i+0, j+0, k+0) * d(i+0, j+0, k+-1))) * __local_m_100);
        }      }    }}{      gridtools::data_view<storage_ijk_t> a= gridtools::make_host_view(a_);
      std::array<int,3> a_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> b= gridtools::make_host_view(b_);
      std::array<int,3> b_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> c= gridtools::make_host_view(c_);
      std::array<int,3> c_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> d= gridtools::make_host_view(d_);
      std::array<int,3> d_offsets{0,0,0};
    for(int k = kMax + -1+0; k >= kMin + 0+0; --k) {
      for(int i = iMin+0; i  <=  iMax+0; ++i) {
        for(int j = jMin+0; j  <=  jMax+0; ++j) {
d(i+0, j+0, k+0) -= (c(i+0, j+0, k+0) * d(i+0, j+0, k+1));
        }      }    }}      a_.sync();
      b_.sync();
      c_.sync();
      d_.sync();
    }
  };
  static constexpr const char* s_name = "tridiagonal_solve_stencil";
  stencil_49 m_stencil_49;
public:

  tridiagonal_solve_stencil(const tridiagonal_solve_stencil&) = delete;

  tridiagonal_solve_stencil(const gridtools::dawn::domain& dom, int rank = 1, int xcols = 1, int ycols = 1) : m_stencil_49(dom, rank, xcols, ycols){
    assert(dom.isize() >= dom.iminus() + dom.iplus());
    assert(dom.jsize() >= dom.jminus() + dom.jplus());
    assert(dom.ksize() >= dom.kminus() + dom.kplus());
    assert(dom.ksize() >= 1);
  }

  void run(storage_ijk_t a, storage_ijk_t b, storage_ijk_t c, storage_ijk_t d) {
    m_stencil_49.run(a,b,c,d);
  }
};
} // namespace cxxnaive
} // namespace dawn_generated

