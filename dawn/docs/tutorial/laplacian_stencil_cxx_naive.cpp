// gtclang (0.0.1-78ef08c-x86_64-linux-gnu-7.4.0)
// based on LLVM/Clang (6.0.0), Dawn (0.0.1)
// Generated on 2019-10-14  15:59:00

#define GRIDTOOLS_CLANG_GENERATED 1
#define GRIDTOOLS_CLANG_BACKEND_T CXXNAIVE
#ifndef BOOST_RESULT_OF_USE_TR1
#define BOOST_RESULT_OF_USE_TR1 1
#endif
#ifndef BOOST_NO_CXX11_DECLTYPE
#define BOOST_NO_CXX11_DECLTYPE 1
#endif
#ifndef GRIDTOOLS_CLANG_HALO_EXTEND
#define GRIDTOOLS_CLANG_HALO_EXTEND 3
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
#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

namespace dawn_generated {
namespace cxxnaive {

struct globals {
  double dx;

  globals() : dx(0) {}
};
} // namespace cxxnaive
} // namespace dawn_generated

namespace dawn_generated {
namespace cxxnaive {

class laplacian_stencil {
private:
  struct stencil_27 {
    // Members

    // Temporary storages
    using tmp_halo_t = gridtools::halo<GRIDTOOLS_CLANG_HALO_EXTEND, GRIDTOOLS_CLANG_HALO_EXTEND, 0>;
    using tmp_meta_data_t = storage_traits_t::storage_info_t<0, 3, tmp_halo_t>;
    using tmp_storage_t = storage_traits_t::data_store_t<float_type, tmp_meta_data_t>;
    const gridtools::clang::domain& m_dom;
    const globals& m_globals;

    // Input/Output storages
    storage_ijk_t& m_out_field;
    storage_ijk_t& m_in_field;

  public:
    stencil_27(const gridtools::clang::domain& dom_, const globals& globals_,
               storage_ijk_t& out_field_, storage_ijk_t& in_field_)
        : m_dom(dom_), m_globals(globals_), m_out_field(out_field_), m_in_field(in_field_) {}

    virtual ~stencil_27() {}

    void sync_storages() {
      m_out_field.sync();
      m_in_field.sync();
    }

    virtual void run(storage_ijk_t& out_field_, storage_ijk_t& in_field_) {
      sync_storages();
      {
        gridtools::data_view<storage_ijk_t> out_field = gridtools::make_host_view(m_out_field);
        std::array<int, 3> out_field_offsets{0, 0, 0};
        gridtools::data_view<storage_ijk_t> in_field = gridtools::make_host_view(m_in_field);
        std::array<int, 3> in_field_offsets{0, 0, 0};
        for(int k = 0 + 0;
            k <= (m_dom.ksize() == 0 ? 0 : (m_dom.ksize() - m_dom.kplus() - 1)) + 0 + 0; ++k) {
          for(int i = m_dom.iminus() + 0; i <= m_dom.isize() - m_dom.iplus() - 1 + 0; ++i) {
            for(int j = m_dom.jminus() + 0; j <= m_dom.jsize() - m_dom.jplus() - 1 + 0; ++j) {
              out_field(i + 0, j + 0, k + 0) = (((((((-(int)4) * in_field(i + 0, j + 0, k + 0)) +
                                                    in_field(i + 1, j + 0, k + 0)) +
                                                   in_field(i + -1, j + 0, k + 0)) +
                                                  in_field(i + 0, j + -1, k + 0)) +
                                                 in_field(i + 0, j + 1, k + 0)) /
                                                (m_globals.dx * m_globals.dx));
            }
          }
        }
      }
      sync_storages();
    }
  };
  static constexpr const char* s_name = "laplacian_stencil";
  globals m_globals;
  stencil_27 m_stencil_27;

public:
  laplacian_stencil(const laplacian_stencil&) = delete;

  laplacian_stencil(const gridtools::clang::domain& dom, storage_ijk_t& out_field,
                    storage_ijk_t& in_field)
      : m_stencil_27(dom, m_globals, out_field, in_field) {}

  // Access-wrapper for globally defined variables

  double get_dx() { return m_globals.dx; }

  void set_dx(double dx) { m_globals.dx = dx; }

  void run(storage_ijk_t out_field, storage_ijk_t in_field) {
    m_stencil_27.run(out_field, in_field);
    ;
  }
};
} // namespace cxxnaive
} // namespace dawn_generated
