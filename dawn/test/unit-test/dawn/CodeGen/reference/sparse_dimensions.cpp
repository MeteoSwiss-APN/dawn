#define DAWN_GENERATED 1
#undef DAWN_BACKEND_T
#define DAWN_BACKEND_T CXXNAIVEICO
#include <driver-includes/unstructured_interface.hpp>


namespace dawn_generated{
namespace cxxnaiveico{
template<typename LibTag>
class sparse_dimensions {
private:

  struct stencil_16 {
    dawn::mesh_t<LibTag> const& m_mesh;
    int m_k_size;
    dawn::edge_field_t<LibTag, double>& m_in;
    dawn::sparse_cell_field_t<LibTag, double>& m_sparse_CE;
    dawn::cell_field_t<LibTag, double>& m_out;
  public:

    stencil_16(dawn::mesh_t<LibTag> const &mesh, int k_size, dawn::edge_field_t<LibTag, double>&in, dawn::sparse_cell_field_t<LibTag, double>&sparse_CE, dawn::cell_field_t<LibTag, double>&out) : m_mesh(mesh), m_k_size(k_size), m_in(in), m_sparse_CE(sparse_CE), m_out(out){}

    ~stencil_16() {
    }

    void sync_storages() {
    }
    static constexpr dawn::driver::unstructured_extent in_extent = {false, 0,0};
    static constexpr dawn::driver::unstructured_extent sparse_CE_extent = {false, 0,0};
    static constexpr dawn::driver::unstructured_extent out_extent = {false, 0,0};

    void run() {
      using dawn::deref;
{
    for(int k = 0+0; k <= ( m_k_size == 0 ? 0 : (m_k_size - 1)) + 0+0; ++k) {
      for(auto const& loc : getCells(LibTag{}, m_mesh)) {
int m_sparse_dimension_idx = 0;
m_out(deref(LibTag{}, loc),k+0) = reduce(LibTag{}, m_mesh,loc, (::dawn::float_type) 1.0, std::vector<dawn::LocationType>{dawn::LocationType::Cells, dawn::LocationType::Edges}, [&](auto& lhs, auto red_loc) { lhs += (m_sparse_CE(deref(LibTag{}, loc),m_sparse_dimension_idx, k+0) * m_in(deref(LibTag{}, red_loc),k+0));
m_sparse_dimension_idx++;
return lhs;
});
      }    }}      sync_storages();
    }
  };
  static constexpr const char* s_name = "sparse_dimensions";
  stencil_16 m_stencil_16;
public:

  sparse_dimensions(const sparse_dimensions&) = delete;

  // Members

  sparse_dimensions(const dawn::mesh_t<LibTag> &mesh, int k_size, dawn::edge_field_t<LibTag, double>& in, dawn::sparse_cell_field_t<LibTag, double>& sparse_CE, dawn::cell_field_t<LibTag, double>& out) : m_stencil_16(mesh, k_size,in,sparse_CE,out){}

  void run() {
    m_stencil_16.run();
;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated

