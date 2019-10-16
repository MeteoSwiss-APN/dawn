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
    storage_ijk_t& m_in;
    storage_ijk_t& m_out;

  public:
    stencil_26(const gridtools::clang::domain& dom_, const globals& globals_, storage_ijk_t& in_,
               storage_ijk_t& out_)
        : m_dom(dom_), m_globals(globals_), m_in(in_), m_out(out_) {}

    virtual ~stencil_26() {}

    void sync_storages() {
      m_in.sync();
      m_out.sync();
    }

    virtual void run(storage_ijk_t& in_, storage_ijk_t& out_) {
      sync_storages();
      {
        gridtools::data_view<storage_ijk_t> in = gridtools::make_host_view(m_in);
        std::array<int, 3> in_offsets{0, 0, 0};
        gridtools::data_view<storage_ijk_t> out = gridtools::make_host_view(m_out);
        std::array<int, 3> out_offsets{0, 0, 0};
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

  laplacian_stencil(const gridtools::clang::domain& dom, storage_ijk_t& in, storage_ijk_t& out)
      : m_stencil_26(dom, m_globals, in, out) {}

  // Access-wrapper for globally defined variables

  double get_dx() { return m_globals.dx; }

  void set_dx(double dx) { m_globals.dx = dx; }

  void run(storage_ijk_t in, storage_ijk_t out) {
    m_stencil_26.run(in, out);
    ;
  }
};
} // namespace cxxnaive
} // namespace dawn_generated
