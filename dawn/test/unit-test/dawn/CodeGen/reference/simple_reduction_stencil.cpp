#define DAWN_GENERATED 1
#undef DAWN_BACKEND_T
#define DAWN_BACKEND_T CXXNAIVEICO
#include <driver-includes/unstructured_interface.hpp>


namespace dawn_generated{
namespace cxxnaiveico{
template<typename LibTag>
class simple_reduction_stencil {
private:

  struct stencil_13 {
    dawn::mesh_t<LibTag> const& m_mesh;
    int m_k_size;
    dawn::cell_field_t<LibTag, double>& m_in;
    dawn::edge_field_t<LibTag, double>& m_out;
  public:

    stencil_13(dawn::mesh_t<LibTag> const &mesh, int k_size, dawn::cell_field_t<LibTag, double>&in, dawn::edge_field_t<LibTag, double>&out) : m_mesh(mesh), m_k_size(k_size), m_in(in), m_out(out){}

    ~stencil_13() {
    }

    void sync_storages() {
    }
    static constexpr dawn::driver::unstructured_extent in_extent = {false, 0,0};
    static constexpr dawn::driver::unstructured_extent out_extent = {false, 0,0};

    void run() {
      using dawn::deref;
{
    for(int k = 0+0; k <= ( m_k_size == 0 ? 0 : (m_k_size - 1)) + 0+0; ++k) {
      for(auto const& loc : getEdges(LibTag{}, m_mesh)) {
int m_sparse_dimension_idx = 0;
m_out(deref(LibTag{}, loc),k+0) = reduceCellToEdge(LibTag{}, m_mesh,loc, (::dawn::float_type) 1.0, [&](auto& lhs, auto const& red_loc) { lhs += m_in(deref(LibTag{}, red_loc),k+0);
m_sparse_dimension_idx++;
return lhs;
});
      }    }}      sync_storages();
    }
  };
  static constexpr const char* s_name = "simple_reduction_stencil";
  stencil_13 m_stencil_13;
public:

  simple_reduction_stencil(const simple_reduction_stencil&) = delete;

  // Members

  simple_reduction_stencil(const dawn::mesh_t<LibTag> &mesh, int k_size, dawn::cell_field_t<LibTag, double>& in, dawn::edge_field_t<LibTag, double>& out) : m_stencil_13(mesh, k_size,in,out){}

  void run() {
    m_stencil_13.run();
;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated

