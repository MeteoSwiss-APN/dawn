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

class global_index_stencil {
private:

  struct stencil_59 {

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

    stencil_59(const gridtools::dawn::domain& dom_, int rank, int xcols, int ycols) : m_dom(dom_), stage120GlobalIIndices({dom_.isize() - dom_.iplus()  + -1 , dom_.isize() - dom_.iplus()  + 0}), stage124GlobalIIndices({dom_.iminus() + 0 , dom_.iminus() + 1}), stage128GlobalJIndices({dom_.jsize() - dom_.jplus()  + -1 , dom_.jsize() - dom_.jplus()  + 0}), stage132GlobalJIndices({dom_.jminus() + 0 , dom_.jminus() + 1}), stage136GlobalIIndices({dom_.iminus() + 0 , dom_.iminus() + 1}), stage136GlobalJIndices({dom_.jminus() + 0 , dom_.jminus() + 1}), stage140GlobalIIndices({dom_.isize() - dom_.iplus()  + -1 , dom_.isize() - dom_.iplus()  + 0}), stage140GlobalJIndices({dom_.jminus() + 0 , dom_.jminus() + 1}), stage144GlobalIIndices({dom_.iminus() + 0 , dom_.iminus() + 1}), stage144GlobalJIndices({dom_.jsize() - dom_.jplus()  + -1 , dom_.jsize() - dom_.jplus()  + 0}), stage148GlobalIIndices({dom_.isize() - dom_.iplus()  + -1 , dom_.isize() - dom_.iplus()  + 0}), stage148GlobalJIndices({dom_.jsize() - dom_.jplus()  + -1 , dom_.jsize() - dom_.jplus()  + 0}), globalOffsets({computeGlobalOffsets(rank, m_dom, xcols, ycols)}){}
    static constexpr dawn::driver::cartesian_extent in_extent = {0,0, 0,0, 0,0};
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
    for(int k = kMin + 0+0; k <= kMax + 0+0; ++k) {
      for(int i = iMin+0; i  <=  iMax+0; ++i) {
        for(int j = jMin+0; j  <=  jMax+0; ++j) {
out(i+0, j+0, k+0) = in(i+0, j+0, k+0);
        }      }      for(int i = iMin+0; i  <=  iMax+0; ++i) {
        for(int j = jMin+0; j  <=  jMax+0; ++j) {
          if(checkOffset(stage120GlobalIIndices[0], stage120GlobalIIndices[1], globalOffsets[0] + i)) {
out(i+0, j+0, k+0) = (::dawn::float_type) 4;
          }        }      }      for(int i = iMin+0; i  <=  iMax+0; ++i) {
        for(int j = jMin+0; j  <=  jMax+0; ++j) {
          if(checkOffset(stage124GlobalIIndices[0], stage124GlobalIIndices[1], globalOffsets[0] + i)) {
out(i+0, j+0, k+0) = (::dawn::float_type) 8;
          }        }      }      for(int i = iMin+0; i  <=  iMax+0; ++i) {
        for(int j = jMin+0; j  <=  jMax+0; ++j) {
          if(checkOffset(stage128GlobalJIndices[0], stage128GlobalJIndices[1], globalOffsets[1] + j)) {
out(i+0, j+0, k+0) = (::dawn::float_type) 6;
          }        }      }      for(int i = iMin+0; i  <=  iMax+0; ++i) {
        for(int j = jMin+0; j  <=  jMax+0; ++j) {
          if(checkOffset(stage132GlobalJIndices[0], stage132GlobalJIndices[1], globalOffsets[1] + j)) {
out(i+0, j+0, k+0) = (::dawn::float_type) 2;
          }        }      }      for(int i = iMin+0; i  <=  iMax+0; ++i) {
        for(int j = jMin+0; j  <=  jMax+0; ++j) {
          if(checkOffset(stage136GlobalIIndices[0], stage136GlobalIIndices[1], globalOffsets[0] + i) && checkOffset(stage136GlobalJIndices[0], stage136GlobalJIndices[1], globalOffsets[1] + j)) {
out(i+0, j+0, k+0) = (::dawn::float_type) 1;
          }        }      }      for(int i = iMin+0; i  <=  iMax+0; ++i) {
        for(int j = jMin+0; j  <=  jMax+0; ++j) {
          if(checkOffset(stage140GlobalIIndices[0], stage140GlobalIIndices[1], globalOffsets[0] + i) && checkOffset(stage140GlobalJIndices[0], stage140GlobalJIndices[1], globalOffsets[1] + j)) {
out(i+0, j+0, k+0) = (::dawn::float_type) 3;
          }        }      }      for(int i = iMin+0; i  <=  iMax+0; ++i) {
        for(int j = jMin+0; j  <=  jMax+0; ++j) {
          if(checkOffset(stage144GlobalIIndices[0], stage144GlobalIIndices[1], globalOffsets[0] + i) && checkOffset(stage144GlobalJIndices[0], stage144GlobalJIndices[1], globalOffsets[1] + j)) {
out(i+0, j+0, k+0) = (::dawn::float_type) 7;
          }        }      }      for(int i = iMin+0; i  <=  iMax+0; ++i) {
        for(int j = jMin+0; j  <=  jMax+0; ++j) {
          if(checkOffset(stage148GlobalIIndices[0], stage148GlobalIIndices[1], globalOffsets[0] + i) && checkOffset(stage148GlobalJIndices[0], stage148GlobalJIndices[1], globalOffsets[1] + j)) {
out(i+0, j+0, k+0) = (::dawn::float_type) 5;
          }        }      }    }}      in_.sync();
      out_.sync();
    }
  };
  static constexpr const char* s_name = "global_index_stencil";
  stencil_59 m_stencil_59;
public:

  global_index_stencil(const global_index_stencil&) = delete;

  global_index_stencil(const gridtools::dawn::domain& dom, int rank = 1, int xcols = 1, int ycols = 1) : m_stencil_59(dom, rank, xcols, ycols){
    assert(dom.isize() >= dom.iminus() + dom.iplus());
    assert(dom.jsize() >= dom.jminus() + dom.jplus());
    assert(dom.ksize() >= dom.kminus() + dom.kplus());
    assert(dom.ksize() >= 1);
  }

  void run(storage_ijk_t in, storage_ijk_t out) {
    m_stencil_59.run(in,out);
  }
};
} // namespace cxxnaive
} // namespace dawn_generated

