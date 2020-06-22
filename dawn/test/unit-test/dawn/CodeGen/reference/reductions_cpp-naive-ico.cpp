#define DAWN_GENERATED 1
#undef DAWN_BACKEND_T
#define DAWN_BACKEND_T CXXNAIVEICO
#include <driver-includes/unstructured_interface.hpp>


namespace dawn_generated{
namespace cxxnaiveico{
template<typename LibTag>
class reductions {
private:

  struct stencil_35 {
    dawn::mesh_t<LibTag> const& m_mesh;
    int m_k_size;
    dawn::edge_field_t<LibTag, double>& m_lhs_field;
    dawn::edge_field_t<LibTag, double>& m_rhs_field;
    dawn::cell_field_t<LibTag, double>& m_cell_field;
    dawn::vertex_field_t<LibTag, double>& m_node_field;
  public:

    stencil_35(dawn::mesh_t<LibTag> const &mesh, int k_size, dawn::edge_field_t<LibTag, double>&lhs_field, dawn::edge_field_t<LibTag, double>&rhs_field, dawn::cell_field_t<LibTag, double>&cell_field, dawn::vertex_field_t<LibTag, double>&node_field) : m_mesh(mesh), m_k_size(k_size), m_lhs_field(lhs_field), m_rhs_field(rhs_field), m_cell_field(cell_field), m_node_field(node_field){}

    ~stencil_35() {
    }

    void sync_storages() {
    }
    static constexpr dawn::driver::unstructured_extent lhs_field_extent = {false, 0,0};
    static constexpr dawn::driver::unstructured_extent rhs_field_extent = {false, 0,0};
    static constexpr dawn::driver::unstructured_extent cell_field_extent = {false, 0,0};
    static constexpr dawn::driver::unstructured_extent node_field_extent = {false, 0,0};

    void run() {
      using dawn::deref;
{
    for(int k = 0+0; k <= ( m_k_size == 0 ? 0 : (m_k_size - 1)) + 0+0; ++k) {
      for(auto const& loc : getCells(LibTag{}, m_mesh)) {
{
int sparse_dimension_idx0 = 0;
m_lhs_field(deref(LibTag{}, loc),k+0) = ((m_rhs_field(deref(LibTag{}, loc),k+0) + reduce(LibTag{}, m_mesh,loc, (::dawn::float_type) 0.000000, std::vector<dawn::LocationType>{dawn::LocationType::Edges, dawn::LocationType::Cells}, [&](auto& lhs, auto red_loc1) { lhs += m_cell_field(deref(LibTag{}, red_loc1),k+0);
sparse_dimension_idx0++;
return lhs;
})) + reduce(LibTag{}, m_mesh,loc, (::dawn::float_type) 0.000000, std::vector<dawn::LocationType>{dawn::LocationType::Edges, dawn::LocationType::Vertices}, [&](auto& lhs, auto red_loc1) { lhs += m_node_field(deref(LibTag{}, red_loc1),k+0);
sparse_dimension_idx0++;
return lhs;
}));
}
      }    }}      sync_storages();
    }
  };
  static constexpr const char* s_name = "reductions";
  stencil_35 m_stencil_35;
public:

  reductions(const reductions&) = delete;

  // Members

  reductions(const dawn::mesh_t<LibTag> &mesh, int k_size, dawn::edge_field_t<LibTag, double>& lhs_field, dawn::edge_field_t<LibTag, double>& rhs_field, dawn::cell_field_t<LibTag, double>& cell_field, dawn::vertex_field_t<LibTag, double>& node_field) : m_stencil_35(mesh, k_size,lhs_field,rhs_field,cell_field,node_field){}

  void run() {
    m_stencil_35.run();
;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
