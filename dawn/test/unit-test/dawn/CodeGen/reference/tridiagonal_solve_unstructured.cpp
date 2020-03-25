#define DAWN_GENERATED 1
#undef DAWN_BACKEND_T
#define DAWN_BACKEND_T CXXNAIVEICO
#include <driver-includes/unstructured_interface.hpp>


namespace dawn_generated{
namespace cxxnaiveico{
template<typename LibTag>
class tridiagonal_solve_unstructured {
private:

  struct stencil_55 {
    dawn::mesh_t<LibTag> const& m_mesh;
    int m_k_size;
    dawn::cell_field_t<LibTag, double>& m_a;
    dawn::cell_field_t<LibTag, double>& m_b;
    dawn::cell_field_t<LibTag, double>& m_c;
    dawn::cell_field_t<LibTag, double>& m_d;
  public:

    stencil_55(dawn::mesh_t<LibTag> const &mesh, int k_size, dawn::cell_field_t<LibTag, double>&a, dawn::cell_field_t<LibTag, double>&b, dawn::cell_field_t<LibTag, double>&c, dawn::cell_field_t<LibTag, double>&d) : m_mesh(mesh), m_k_size(k_size), m_a(a), m_b(b), m_c(c), m_d(d){}

    ~stencil_55() {
    }

    void sync_storages() {
    }
    static constexpr dawn::driver::unstructured_extent a_extent = {false, 0,0};
    static constexpr dawn::driver::unstructured_extent b_extent = {false, 0,0};
    static constexpr dawn::driver::unstructured_extent c_extent = {false, -1,0};
    static constexpr dawn::driver::unstructured_extent d_extent = {false, -1,1};

    void run() {
      using dawn::deref;
{
    for(int k = 0+0; k <= 0+0; ++k) {
      for(auto const& loc : getCells(LibTag{}, m_mesh)) {
m_d(deref(LibTag{}, loc),k+0) = (m_d(deref(LibTag{}, loc),k+0) / m_b(deref(LibTag{}, loc),k+0));
      }      for(auto const& loc : getCells(LibTag{}, m_mesh)) {
m_c(deref(LibTag{}, loc),k+0) = (m_c(deref(LibTag{}, loc),k+0) / m_b(deref(LibTag{}, loc),k+0));
      }      for(auto const& loc : getCells(LibTag{}, m_mesh)) {
      }    }    for(int k = 1+0; k <= ( m_k_size == 0 ? 0 : (m_k_size - 1)) + 0+0; ++k) {
      for(auto const& loc : getCells(LibTag{}, m_mesh)) {
      }      for(auto const& loc : getCells(LibTag{}, m_mesh)) {
m___tmp_m_112(deref(LibTag{}, loc),k+0) = ((::dawn::float_type) 1.0 / (m_b(deref(LibTag{}, loc),k+0) - (m_a(deref(LibTag{}, loc),k+0) * m_c(deref(LibTag{}, loc),k+-1))));
      }      for(auto const& loc : getCells(LibTag{}, m_mesh)) {
m_d(deref(LibTag{}, loc),k+0) = ((m_d(deref(LibTag{}, loc),k+0) - (m_a(deref(LibTag{}, loc),k+0) * m_d(deref(LibTag{}, loc),k+-1))) * m___tmp_m_112(deref(LibTag{}, loc),k+0));
      }    }}{
    for(int k = 1+0; k <= ( m_k_size == 0 ? 0 : (m_k_size - 1)) + 0+0; ++k) {
      for(auto const& loc : getCells(LibTag{}, m_mesh)) {
m_c(deref(LibTag{}, loc),k+0) = (m_c(deref(LibTag{}, loc),k+0) * m___tmp_m_112(deref(LibTag{}, loc),k+0));
      }    }}{
    for(int k = ( m_k_size == 0 ? 0 : (m_k_size - 1)) + -1+0; k >= 0+0; --k) {
      for(auto const& loc : getCells(LibTag{}, m_mesh)) {
m_d(deref(LibTag{}, loc),k+0) -= (m_c(deref(LibTag{}, loc),k+0) * m_d(deref(LibTag{}, loc),k+1));
      }    }}      sync_storages();
    }
  };
  static constexpr const char* s_name = "tridiagonal_solve_unstructured";
  stencil_55 m_stencil_55;
public:

  tridiagonal_solve_unstructured(const tridiagonal_solve_unstructured&) = delete;

  // Members

  tridiagonal_solve_unstructured(const dawn::mesh_t<LibTag> &mesh, int k_size, dawn::cell_field_t<LibTag, double>& a, dawn::cell_field_t<LibTag, double>& b, dawn::cell_field_t<LibTag, double>& c, dawn::cell_field_t<LibTag, double>& d) : m_stencil_55(mesh, k_size,a,b,c,d){}

  void run() {
    m_stencil_55.run();
;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated

