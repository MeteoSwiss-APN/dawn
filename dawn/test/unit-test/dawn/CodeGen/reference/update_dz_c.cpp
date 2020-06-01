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

struct globals {
  double dt;

  globals() : dt(0){
  }
};
} // namespace cxxnaive
} // namespace dawn_generated


namespace dawn_generated{
namespace cxxnaive{

class update_dz_c {
private:

  struct stencil_443 {

    // Members

    // Temporary storages
    using tmp_halo_t = gridtools::halo< GRIDTOOLS_DAWN_HALO_EXTENT, GRIDTOOLS_DAWN_HALO_EXTENT, 0>;
    using tmp_meta_data_t = storage_traits_t::storage_info_t< 0, 3, tmp_halo_t >;
    using tmp_storage_t = storage_traits_t::data_store_t< ::dawn::float_type, tmp_meta_data_t>;
    const gridtools::dawn::domain m_dom;
    const globals& m_globals;

    // Input/Output storages
    tmp_meta_data_t m_tmp_meta_data;
    tmp_storage_t m_xfx;
    tmp_storage_t m_yfx;
    tmp_storage_t m_fx;
    tmp_storage_t m_fy;
  public:

    stencil_443(const gridtools::dawn::domain& dom_, const globals& globals_, int rank, int xcols, int ycols) : m_dom(dom_), m_globals(globals_), m_tmp_meta_data(dom_.isize() + 1, dom_.jsize() + 1, dom_.ksize() + 2*0), m_xfx(m_tmp_meta_data), m_yfx(m_tmp_meta_data), m_fx(m_tmp_meta_data), m_fy(m_tmp_meta_data){}
    static constexpr dawn::driver::cartesian_extent dp_ref_extent = {0,1, 0,1, -2,1};
    static constexpr dawn::driver::cartesian_extent zs_extent = {0,0, 0,0, 0,0};
    static constexpr dawn::driver::cartesian_extent area_extent = {0,0, 0,0, 0,0};
    static constexpr dawn::driver::cartesian_extent ut_extent = {0,1, 0,1, -2,1};
    static constexpr dawn::driver::cartesian_extent vt_extent = {0,1, 0,1, -2,1};
    static constexpr dawn::driver::cartesian_extent gz_extent = {0,0, 0,0, 0,0};
    static constexpr dawn::driver::cartesian_extent gz_x_extent = {-1,1, 0,1, 0,0};
    static constexpr dawn::driver::cartesian_extent gz_y_extent = {0,1, -1,1, 0,0};
    static constexpr dawn::driver::cartesian_extent ws3_extent = {0,0, 0,0, 0,0};
    static constexpr dawn::driver::cartesian_extent gz_0_extent = {0,0, 0,0, 0,1};

    void run(storage_ijk_t& dp_ref_, storage_ijk_t& zs_, storage_ijk_t& area_, storage_ijk_t& ut_, storage_ijk_t& vt_, storage_ijk_t& gz_, storage_ijk_t& gz_x_, storage_ijk_t& gz_y_, storage_ijk_t& ws3_, storage_ijk_t& gz_0_) {
      int iMin = m_dom.iminus();
      int iMax = m_dom.isize() - m_dom.iplus() - 1;
      int jMin = m_dom.jminus();
      int jMax = m_dom.jsize() - m_dom.jplus() - 1;
      int kMin = m_dom.kminus();
      int kMax = m_dom.ksize() - m_dom.kplus() - 1;
      dp_ref_.sync();
      zs_.sync();
      area_.sync();
      ut_.sync();
      vt_.sync();
      gz_.sync();
      gz_x_.sync();
      gz_y_.sync();
      ws3_.sync();
      gz_0_.sync();
{      gridtools::data_view<storage_ijk_t> dp_ref= gridtools::make_host_view(dp_ref_);
      std::array<int,3> dp_ref_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> zs= gridtools::make_host_view(zs_);
      std::array<int,3> zs_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> area= gridtools::make_host_view(area_);
      std::array<int,3> area_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> ut= gridtools::make_host_view(ut_);
      std::array<int,3> ut_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> vt= gridtools::make_host_view(vt_);
      std::array<int,3> vt_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz= gridtools::make_host_view(gz_);
      std::array<int,3> gz_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_x= gridtools::make_host_view(gz_x_);
      std::array<int,3> gz_x_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_y= gridtools::make_host_view(gz_y_);
      std::array<int,3> gz_y_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> ws3= gridtools::make_host_view(ws3_);
      std::array<int,3> ws3_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_0= gridtools::make_host_view(gz_0_);
      std::array<int,3> gz_0_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> xfx= gridtools::make_host_view(m_xfx);
      std::array<int,3> xfx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> yfx= gridtools::make_host_view(m_yfx);
      std::array<int,3> yfx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> fx= gridtools::make_host_view(m_fx);
      std::array<int,3> fx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> fy= gridtools::make_host_view(m_fy);
      std::array<int,3> fy_offsets{0,0,0};
    for(int k = kMin + 0+0; k <= kMin + 1+0; ++k) {
      for(int i = iMin+0; i  <=  iMax+1; ++i) {
        for(int j = jMin+0; j  <=  jMax+1; ++j) {
::dawn::float_type __local_ratio__d15_16_429 = (dp_ref(i+0, j+0, k+0) / (dp_ref(i+0, j+0, k+0) + dp_ref(i+0, j+0, k+1)));
xfx(i+0, j+0, k+0) = (ut(i+0, j+0, k+0) + ((ut(i+0, j+0, k+0) - ut(i+0, j+0, k+1)) * __local_ratio__d15_16_429));
::dawn::float_type __local_ratio__d15_17_431 = (dp_ref(i+0, j+0, k+0) / (dp_ref(i+0, j+0, k+0) + dp_ref(i+0, j+0, k+1)));
yfx(i+0, j+0, k+0) = (vt(i+0, j+0, k+0) + ((vt(i+0, j+0, k+0) - vt(i+0, j+0, k+1)) * __local_ratio__d15_17_431));
        }      }    }}{      gridtools::data_view<storage_ijk_t> dp_ref= gridtools::make_host_view(dp_ref_);
      std::array<int,3> dp_ref_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> zs= gridtools::make_host_view(zs_);
      std::array<int,3> zs_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> area= gridtools::make_host_view(area_);
      std::array<int,3> area_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> ut= gridtools::make_host_view(ut_);
      std::array<int,3> ut_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> vt= gridtools::make_host_view(vt_);
      std::array<int,3> vt_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz= gridtools::make_host_view(gz_);
      std::array<int,3> gz_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_x= gridtools::make_host_view(gz_x_);
      std::array<int,3> gz_x_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_y= gridtools::make_host_view(gz_y_);
      std::array<int,3> gz_y_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> ws3= gridtools::make_host_view(ws3_);
      std::array<int,3> ws3_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_0= gridtools::make_host_view(gz_0_);
      std::array<int,3> gz_0_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> xfx= gridtools::make_host_view(m_xfx);
      std::array<int,3> xfx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> yfx= gridtools::make_host_view(m_yfx);
      std::array<int,3> yfx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> fx= gridtools::make_host_view(m_fx);
      std::array<int,3> fx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> fy= gridtools::make_host_view(m_fy);
      std::array<int,3> fy_offsets{0,0,0};
    for(int k = kMax + -1+0; k <= kMax + 0+0; ++k) {
      for(int i = iMin+0; i  <=  iMax+1; ++i) {
        for(int j = jMin+0; j  <=  jMax+1; ++j) {
::dawn::float_type __local_ratio__c74_19_433 = (dp_ref(i+0, j+0, k+-1) / (dp_ref(i+0, j+0, k+-2) + dp_ref(i+0, j+0, k+-1)));
xfx(i+0, j+0, k+0) = (ut(i+0, j+0, k+-1) + ((ut(i+0, j+0, k+-1) - ut(i+0, j+0, k+-2)) * __local_ratio__c74_19_433));
::dawn::float_type __local_ratio__c74_20_434 = (dp_ref(i+0, j+0, k+-1) / (dp_ref(i+0, j+0, k+-2) + dp_ref(i+0, j+0, k+-1)));
yfx(i+0, j+0, k+0) = (vt(i+0, j+0, k+-1) + ((vt(i+0, j+0, k+-1) - vt(i+0, j+0, k+-2)) * __local_ratio__c74_20_434));
        }      }    }}{      gridtools::data_view<storage_ijk_t> dp_ref= gridtools::make_host_view(dp_ref_);
      std::array<int,3> dp_ref_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> zs= gridtools::make_host_view(zs_);
      std::array<int,3> zs_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> area= gridtools::make_host_view(area_);
      std::array<int,3> area_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> ut= gridtools::make_host_view(ut_);
      std::array<int,3> ut_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> vt= gridtools::make_host_view(vt_);
      std::array<int,3> vt_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz= gridtools::make_host_view(gz_);
      std::array<int,3> gz_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_x= gridtools::make_host_view(gz_x_);
      std::array<int,3> gz_x_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_y= gridtools::make_host_view(gz_y_);
      std::array<int,3> gz_y_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> ws3= gridtools::make_host_view(ws3_);
      std::array<int,3> ws3_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_0= gridtools::make_host_view(gz_0_);
      std::array<int,3> gz_0_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> xfx= gridtools::make_host_view(m_xfx);
      std::array<int,3> xfx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> yfx= gridtools::make_host_view(m_yfx);
      std::array<int,3> yfx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> fx= gridtools::make_host_view(m_fx);
      std::array<int,3> fx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> fy= gridtools::make_host_view(m_fy);
      std::array<int,3> fy_offsets{0,0,0};
    for(int k = kMin + 1+0; k <= kMax + -1+0; ++k) {
      for(int i = iMin+0; i  <=  iMax+1; ++i) {
        for(int j = jMin+0; j  <=  jMax+1; ++j) {
::dawn::float_type __local_int_ratio__330_22_435 = ((::dawn::float_type) 1.0 / (dp_ref(i+0, j+0, k+-1) + dp_ref(i+0, j+0, k+0)));
xfx(i+0, j+0, k+0) = (((dp_ref(i+0, j+0, k+0) * ut(i+0, j+0, k+-1)) + (dp_ref(i+0, j+0, k+-1) * ut(i+0, j+0, k+0))) * __local_int_ratio__330_22_435);
::dawn::float_type __local_int_ratio__330_23_436 = ((::dawn::float_type) 1.0 / (dp_ref(i+0, j+0, k+-1) + dp_ref(i+0, j+0, k+0)));
yfx(i+0, j+0, k+0) = (((dp_ref(i+0, j+0, k+0) * vt(i+0, j+0, k+-1)) + (dp_ref(i+0, j+0, k+-1) * vt(i+0, j+0, k+0))) * __local_int_ratio__330_23_436);
        }      }    }}{      gridtools::data_view<storage_ijk_t> dp_ref= gridtools::make_host_view(dp_ref_);
      std::array<int,3> dp_ref_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> zs= gridtools::make_host_view(zs_);
      std::array<int,3> zs_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> area= gridtools::make_host_view(area_);
      std::array<int,3> area_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> ut= gridtools::make_host_view(ut_);
      std::array<int,3> ut_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> vt= gridtools::make_host_view(vt_);
      std::array<int,3> vt_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz= gridtools::make_host_view(gz_);
      std::array<int,3> gz_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_x= gridtools::make_host_view(gz_x_);
      std::array<int,3> gz_x_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_y= gridtools::make_host_view(gz_y_);
      std::array<int,3> gz_y_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> ws3= gridtools::make_host_view(ws3_);
      std::array<int,3> ws3_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_0= gridtools::make_host_view(gz_0_);
      std::array<int,3> gz_0_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> xfx= gridtools::make_host_view(m_xfx);
      std::array<int,3> xfx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> yfx= gridtools::make_host_view(m_yfx);
      std::array<int,3> yfx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> fx= gridtools::make_host_view(m_fx);
      std::array<int,3> fx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> fy= gridtools::make_host_view(m_fy);
      std::array<int,3> fy_offsets{0,0,0};
    for(int k = kMin + 0+0; k <= kMax + 0+0; ++k) {
      for(int i = iMin+0; i  <=  iMax+1; ++i) {
        for(int j = jMin+0; j  <=  jMax+1; ++j) {
::dawn::float_type __local_fx__389_25_437 = (xfx(i+0, j+0, k+0) * ((xfx(i+0, j+0, k+0) > (::dawn::float_type) 0.0) ? gz_x(i+-1, j+0, k+0) : gz_x(i+0, j+0, k+0)));
::dawn::float_type __local_fy__389_25_438 = (yfx(i+0, j+0, k+0) * ((yfx(i+0, j+0, k+0) > (::dawn::float_type) 0.0) ? gz_y(i+0, j+-1, k+0) : gz_y(i+0, j+0, k+0)));
fx(i+0, j+0, k+0) = __local_fx__389_25_437;
fy(i+0, j+0, k+0) = __local_fy__389_25_438;
        }      }      for(int i = iMin+0; i  <=  iMax+0; ++i) {
        for(int j = jMin+0; j  <=  jMax+0; ++j) {
gz_0(i+0, j+0, k+0) = ((((((gz_y(i+0, j+0, k+0) * area(i+0, j+0, k+0)) + fx(i+0, j+0, k+0)) - fx(i+1, j+0, k+0)) + fy(i+0, j+0, k+0)) - fy(i+0, j+1, k+0)) / ((((area(i+0, j+0, k+0) + xfx(i+0, j+0, k+0)) - xfx(i+1, j+0, k+0)) + yfx(i+0, j+0, k+0)) - yfx(i+0, j+1, k+0)));
        }      }    }}{      gridtools::data_view<storage_ijk_t> dp_ref= gridtools::make_host_view(dp_ref_);
      std::array<int,3> dp_ref_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> zs= gridtools::make_host_view(zs_);
      std::array<int,3> zs_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> area= gridtools::make_host_view(area_);
      std::array<int,3> area_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> ut= gridtools::make_host_view(ut_);
      std::array<int,3> ut_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> vt= gridtools::make_host_view(vt_);
      std::array<int,3> vt_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz= gridtools::make_host_view(gz_);
      std::array<int,3> gz_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_x= gridtools::make_host_view(gz_x_);
      std::array<int,3> gz_x_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_y= gridtools::make_host_view(gz_y_);
      std::array<int,3> gz_y_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> ws3= gridtools::make_host_view(ws3_);
      std::array<int,3> ws3_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_0= gridtools::make_host_view(gz_0_);
      std::array<int,3> gz_0_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> xfx= gridtools::make_host_view(m_xfx);
      std::array<int,3> xfx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> yfx= gridtools::make_host_view(m_yfx);
      std::array<int,3> yfx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> fx= gridtools::make_host_view(m_fx);
      std::array<int,3> fx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> fy= gridtools::make_host_view(m_fy);
      std::array<int,3> fy_offsets{0,0,0};
    for(int k = kMax + -1+0; k <= kMax + 0+0; ++k) {
      for(int i = iMin+0; i  <=  iMax+0; ++i) {
        for(int j = jMin+0; j  <=  jMax+0; ++j) {
gz_0(i+0, j+0, k+0) = gz(i+0, j+0, k+0);
        }      }    }}{      gridtools::data_view<storage_ijk_t> dp_ref= gridtools::make_host_view(dp_ref_);
      std::array<int,3> dp_ref_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> zs= gridtools::make_host_view(zs_);
      std::array<int,3> zs_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> area= gridtools::make_host_view(area_);
      std::array<int,3> area_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> ut= gridtools::make_host_view(ut_);
      std::array<int,3> ut_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> vt= gridtools::make_host_view(vt_);
      std::array<int,3> vt_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz= gridtools::make_host_view(gz_);
      std::array<int,3> gz_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_x= gridtools::make_host_view(gz_x_);
      std::array<int,3> gz_x_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_y= gridtools::make_host_view(gz_y_);
      std::array<int,3> gz_y_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> ws3= gridtools::make_host_view(ws3_);
      std::array<int,3> ws3_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_0= gridtools::make_host_view(gz_0_);
      std::array<int,3> gz_0_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> xfx= gridtools::make_host_view(m_xfx);
      std::array<int,3> xfx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> yfx= gridtools::make_host_view(m_yfx);
      std::array<int,3> yfx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> fx= gridtools::make_host_view(m_fx);
      std::array<int,3> fx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> fy= gridtools::make_host_view(m_fy);
      std::array<int,3> fy_offsets{0,0,0};
    for(int k = kMax + -1+0; k <= kMax + 0+0; ++k) {
      for(int i = iMin+0; i  <=  iMax+0; ++i) {
        for(int j = jMin+0; j  <=  jMax+0; ++j) {
ws3(i+0, j+0, k+0) = ((zs(i+0, j+0, k+0) - gz_0(i+0, j+0, k+0)) * ((::dawn::float_type) 1.0 / m_globals.dt));
        }      }    }}{      gridtools::data_view<storage_ijk_t> dp_ref= gridtools::make_host_view(dp_ref_);
      std::array<int,3> dp_ref_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> zs= gridtools::make_host_view(zs_);
      std::array<int,3> zs_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> area= gridtools::make_host_view(area_);
      std::array<int,3> area_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> ut= gridtools::make_host_view(ut_);
      std::array<int,3> ut_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> vt= gridtools::make_host_view(vt_);
      std::array<int,3> vt_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz= gridtools::make_host_view(gz_);
      std::array<int,3> gz_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_x= gridtools::make_host_view(gz_x_);
      std::array<int,3> gz_x_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_y= gridtools::make_host_view(gz_y_);
      std::array<int,3> gz_y_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> ws3= gridtools::make_host_view(ws3_);
      std::array<int,3> ws3_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> gz_0= gridtools::make_host_view(gz_0_);
      std::array<int,3> gz_0_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> xfx= gridtools::make_host_view(m_xfx);
      std::array<int,3> xfx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> yfx= gridtools::make_host_view(m_yfx);
      std::array<int,3> yfx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> fx= gridtools::make_host_view(m_fx);
      std::array<int,3> fx_offsets{0,0,0};
      gridtools::data_view<tmp_storage_t> fy= gridtools::make_host_view(m_fy);
      std::array<int,3> fy_offsets{0,0,0};
    for(int k = kMin + 0+0; k <= kMax + -1+0; ++k) {
      for(int i = iMin+0; i  <=  iMax+0; ++i) {
        for(int j = jMin+0; j  <=  jMax+0; ++j) {
::dawn::float_type __local_gz_442 = (gz_0(i+0, j+0, k+1) + (::dawn::float_type) 2.0);
gz(i+0, j+0, k+0) = ((gz(i+0, j+0, k+0) > __local_gz_442) ? gz(i+0, j+0, k+0) : __local_gz_442);
        }      }    }}      dp_ref_.sync();
      zs_.sync();
      area_.sync();
      ut_.sync();
      vt_.sync();
      gz_.sync();
      gz_x_.sync();
      gz_y_.sync();
      ws3_.sync();
      gz_0_.sync();
    }
  };
  static constexpr const char* s_name = "update_dz_c";
  globals m_globals;
  stencil_443 m_stencil_443;
public:

  update_dz_c(const update_dz_c&) = delete;

  // Members
  gridtools::dawn::meta_data_t m_meta_data;
  gridtools::dawn::storage_t m_gz_0;

  update_dz_c(const gridtools::dawn::domain& dom, int rank = 1, int xcols = 1, int ycols = 1) : m_stencil_443(dom, m_globals, rank, xcols, ycols), m_meta_data(dom.isize(), dom.jsize(), dom.ksize() /*+ 2 *0*/ + 1), m_gz_0 (m_meta_data, "gz_0"){
    assert(dom.isize() >= dom.iminus() + dom.iplus());
    assert(dom.jsize() >= dom.jminus() + dom.jplus());
    assert(dom.ksize() >= dom.kminus() + dom.kplus());
    assert(dom.ksize() >= 1);
  }

  // Access-wrapper for globally defined variables

  double get_dt() {
    return m_globals.dt;
  }

  void set_dt(double dt) {
    m_globals.dt=dt;
  }

  void run(storage_ijk_t dp_ref, storage_ijk_t zs, storage_ijk_t area, storage_ijk_t ut, storage_ijk_t vt, storage_ijk_t gz, storage_ijk_t gz_x, storage_ijk_t gz_y, storage_ijk_t ws3) {
    m_stencil_443.run(dp_ref,zs,area,ut,vt,gz,gz_x,gz_y,ws3,m_gz_0);
  }
};
} // namespace cxxnaive
} // namespace dawn_generated
