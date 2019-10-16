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
  struct stencil_26 {

    // Members

    // Temporary storages
    using tmp_halo_t = gridtools::halo<GRIDTOOLS_CLANG_HALO_EXTEND, GRIDTOOLS_CLANG_HALO_EXTEND, 0>;
    using tmp_meta_data_t = storage_traits_t::storage_info_t<0, 3, tmp_halo_t>;
    using tmp_storage_t = storage_traits_t::data_store_t<float_type, tmp_meta_data_t>;
    const gridtools::clang::domain& m_dom;
    const globals& m_globals;

    // Input/Output storages
    storage_ijk_t& m_out;
    storage_ijk_t& m_in;

  public:
    stencil_26(const gridtools::clang::domain& dom_, const globals& globals_, storage_ijk_t& out_,
               storage_ijk_t& in_)
        : m_dom(dom_), m_globals(globals_), m_out(out_), m_in(in_) {}

    virtual ~stencil_26() {}

    void sync_storages() {
      m_out.sync();
      m_in.sync();
    }

    virtual void run(storage_ijk_t& out_, storage_ijk_t& in_) {
      sync_storages();
      {
        gridtools::data_view<storage_ijk_t> out = gridtools::make_host_view(m_out);
        std::array<int, 3> out_offsets{0, 0, 0};
        gridtools::data_view<storage_ijk_t> in = gridtools::make_host_view(m_in);
        std::array<int, 3> in_offsets{0, 0, 0};
        for(int k = 0 + 0;
            k <= (m_dom.ksize() == 0 ? 0 : (m_dom.ksize() - m_dom.kplus() - 1)) + 0 + 0; ++k) {
          for(int i = m_dom.iminus() + 0; i <= m_dom.isize() - m_dom.iplus() - 1 + 0; ++i) {
            for(int j = m_dom.jminus() + 0; j <= m_dom.jsize() - m_dom.jplus() - 1 + 0; ++j) {
              out(i + 0, j + 0, k + 0) =
                  (((in(i + 0, j + 0, k + 0) * (gridtools::clang::float_type)-4.0) +
                    (in(i + 1, j + 0, k + 0) +
                     (in(i + -1, j + 0, k + 0) +
                      (in(i + 0, j + 1, k + 0) + in(i + 0, j + -1, k + 0))))) /
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
  stencil_26 m_stencil_26;

public:
  laplacian_stencil(const laplacian_stencil&) = delete;

  laplacian_stencil(const gridtools::clang::domain& dom, storage_ijk_t& out, storage_ijk_t& in)
      : m_stencil_26(dom, m_globals, out, in) {}

  // Access-wrapper for globally defined variables

  double get_dx() { return m_globals.dx; }

  void set_dx(double dx) { m_globals.dx = dx; }

  void run(storage_ijk_t out, storage_ijk_t in) {
    m_stencil_26.run(out, in);
    ;
  }
};
} // namespace cxxnaive
} // namespace dawn_generated
