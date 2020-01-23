#define DAWN_GENERATED 1
#define DAWN_BACKEND_T CXXNAIVE
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
namespace cxxnaive{

class generated {
private:

  struct stencil_28 {

    // Members
    std::array<int, 2> stage14GlobalJIndices;
    std::array<unsigned int, 2> globalOffsets;

    static std::array<unsigned int, 2> computeGlobalOffsets(int rank, const gridtools::dawn::domain& dom, int xcols, int ycols) {
      unsigned int rankOnDefaultFace = rank % (xcols * ycols);
      unsigned int row = rankOnDefaultFace / xcols;
      unsigned int col = rankOnDefaultFace % ycols;
      return {col * (dom.isize() - dom.iplus()), row * (dom.jsize() - dom.jplus())};
    }

    static bool checkOffset(unsigned int min, unsigned int max, unsigned int val) {
      return (min <= val && val < max);
    }

    // Temporary storages
    using tmp_halo_t = gridtools::halo< GRIDTOOLS_DAWN_HALO_EXTENT, GRIDTOOLS_DAWN_HALO_EXTENT, 0>;
    using tmp_meta_data_t = storage_traits_t::storage_info_t< 0, 3, tmp_halo_t >;
    using tmp_storage_t = storage_traits_t::data_store_t< ::dawn::float_type, tmp_meta_data_t>;
    const gridtools::dawn::domain m_dom;

    // Input/Output storages
  public:

    stencil_28(const gridtools::dawn::domain& dom_, int rank, int xcols, int ycols) : m_dom(dom_), stage14GlobalJIndices({dom_.jminus() + 0 , dom_.jminus() + 2}), globalOffsets({computeGlobalOffsets(rank, m_dom, xcols, ycols)}){}

    void run(storage_ijk_t& in_field_, storage_ijk_t& out_field_) {
      int iMin = m_dom.iminus();
      int iMax = m_dom.isize() - m_dom.iplus() - 1;
      int jMin = m_dom.jminus();
      int jMax = m_dom.jsize() - m_dom.jplus() - 1;
      int kMin = m_dom.kminus();
      int kMax = m_dom.ksize() - m_dom.kplus() - 1;
      in_field_.sync();
      out_field_.sync();
{      gridtools::data_view<storage_ijk_t> in_field= gridtools::make_host_view(in_field_);
      std::array<int,3> in_field_offsets{0,0,0};
      gridtools::data_view<storage_ijk_t> out_field= gridtools::make_host_view(out_field_);
      std::array<int,3> out_field_offsets{0,0,0};
    for(int k = kMin + 0+0; k <= kMax + 0+0; ++k) {
      for(int i = iMin+0; i  <=  iMax+0; ++i) {
        for(int j = jMin+0; j  <=  jMax+0; ++j) {
{
  out_field(i+0, j+0, k+0) = in_field(i+0, j+0, k+0);
}
        }      }      for(int i = iMin+0; i  <=  iMax+0; ++i) {
        for(int j = jMin+0; j  <=  jMax+0; ++j) {
          if(checkOffset(stage14GlobalJIndices[0], stage14GlobalJIndices[1], globalOffsets[1] + j)) {
{
  out_field(i+0, j+0, k+0) = (int) 10;
}
          }        }      }    }}      in_field_.sync();
      out_field_.sync();
    }
  };
  static constexpr const char* s_name = "generated";
  stencil_28 m_stencil_28;
public:

  generated(const generated&) = delete;

  generated(const gridtools::dawn::domain& dom, int rank = 1, int xcols = 1, int ycols = 1) : m_stencil_28(dom, rank, xcols, ycols){
    assert(dom.isize() >= dom.iminus() + dom.iplus());
    assert(dom.jsize() >= dom.jminus() + dom.jplus());
    assert(dom.ksize() >= dom.kminus() + dom.kplus());
    assert(dom.ksize() >= 1);
  }

  void run(storage_ijk_t in_field, storage_ijk_t out_field) {
    m_stencil_28.run(in_field,out_field);
  }
};
} // namespace cxxnaive
} // namespace dawn_generated
