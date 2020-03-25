//---- Preprocessor defines ----
#define DAWN_GENERATED 1
#undef DAWN_BACKEND_T
#define DAWN_BACKEND_T CXXNAIVEICO
#include <driver-includes/unstructured_interface.hpp>

//---- Includes ----
#include "driver-includes/gridtools_includes.hpp"
using namespace gridtools::dawn;

//---- Globals ----

//---- Stencils ----
namespace dawn_generated{
namespace cxxnaiveico{
template<typename LibTag>
class ICON_laplacian_stencil {
private:

  struct stencil_64 {
    dawn::mesh_t<LibTag> const& m_mesh;
    int m_k_size;
    dawn::edge_field_t<LibTag, double>& m_vec;
    dawn::cell_field_t<LibTag, double>& m_div_vec;
    dawn::vertex_field_t<LibTag, double>& m_rot_vec;
    dawn::edge_field_t<LibTag, double>& m_nabla2t1_vec;
    dawn::edge_field_t<LibTag, double>& m_nabla2t2_vec;
    dawn::edge_field_t<LibTag, double>& m_nabla2_vec;
    dawn::edge_field_t<LibTag, double>& m_primal_edge_length;
    dawn::edge_field_t<LibTag, double>& m_dual_edge_length;
    dawn::edge_field_t<LibTag, double>& m_tangent_orientation;
    dawn::sparse_vertex_field_t<LibTag, double>& m_geofac_rot;
    dawn::sparse_cell_field_t<LibTag, double>& m_geofac_div;
  public:

    stencil_64(dawn::mesh_t<LibTag> const &mesh, int k_size, dawn::edge_field_t<LibTag, double>&vec, dawn::cell_field_t<LibTag, double>&div_vec, dawn::vertex_field_t<LibTag, double>&rot_vec, dawn::edge_field_t<LibTag, double>&nabla2t1_vec, dawn::edge_field_t<LibTag, double>&nabla2t2_vec, dawn::edge_field_t<LibTag, double>&nabla2_vec, dawn::edge_field_t<LibTag, double>&primal_edge_length, dawn::edge_field_t<LibTag, double>&dual_edge_length, dawn::edge_field_t<LibTag, double>&tangent_orientation, dawn::sparse_vertex_field_t<LibTag, double>&geofac_rot, dawn::sparse_cell_field_t<LibTag, double>&geofac_div) : m_mesh(mesh), m_k_size(k_size), m_vec(vec), m_div_vec(div_vec), m_rot_vec(rot_vec), m_nabla2t1_vec(nabla2t1_vec), m_nabla2t2_vec(nabla2t2_vec), m_nabla2_vec(nabla2_vec), m_primal_edge_length(primal_edge_length), m_dual_edge_length(dual_edge_length), m_tangent_orientation(tangent_orientation), m_geofac_rot(geofac_rot), m_geofac_div(geofac_div){}

    ~stencil_64() {
    }

    void sync_storages() {
    }
    static constexpr dawn::driver::unstructured_extent vec_extent = {false, 0,0};
    static constexpr dawn::driver::unstructured_extent div_vec_extent = {false, 0,0};
    static constexpr dawn::driver::unstructured_extent rot_vec_extent = {false, 0,0};
    static constexpr dawn::driver::unstructured_extent nabla2t1_vec_extent = {false, 0,0};
    static constexpr dawn::driver::unstructured_extent nabla2t2_vec_extent = {false, 0,0};
    static constexpr dawn::driver::unstructured_extent nabla2_vec_extent = {false, 0,0};
    static constexpr dawn::driver::unstructured_extent primal_edge_length_extent = {false, 0,0};
    static constexpr dawn::driver::unstructured_extent dual_edge_length_extent = {false, 0,0};
    static constexpr dawn::driver::unstructured_extent tangent_orientation_extent = {false, 0,0};
    static constexpr dawn::driver::unstructured_extent geofac_rot_extent = {false, 0,0};
    static constexpr dawn::driver::unstructured_extent geofac_div_extent = {false, 0,0};

    void run() {
      using dawn::deref;
{
    for(int k = 0+0; k <= ( m_k_size == 0 ? 0 : (m_k_size - 1)) + 0+0; ++k) {
      for(auto const& loc : getCells(LibTag{}, m_mesh)) {
int m_sparse_dimension_idx = 0;
m_div_vec(deref(LibTag{}, loc),k+0) = reduce(LibTag{}, m_mesh,loc, (::dawn::float_type) 0.0, std::vector<dawn::LocationType>{dawn::LocationType::Cells, dawn::LocationType::Edges}, [&](auto& lhs, auto red_loc) { lhs += (m_vec(deref(LibTag{}, red_loc),k+0) * m_geofac_div(deref(LibTag{}, loc),m_sparse_dimension_idx, k+0));
m_sparse_dimension_idx++;
return lhs;
});
      }      for(auto const& loc : getEdges(LibTag{}, m_mesh)) {
int m_sparse_dimension_idx = 0;
m_nabla2t2_vec(deref(LibTag{}, loc),k+0) = reduce(LibTag{}, m_mesh,loc, (::dawn::float_type) 0.0, std::vector<dawn::LocationType>{dawn::LocationType::Edges, dawn::LocationType::Cells}, [&](auto& lhs, auto red_loc, auto const& weight) {
lhs += weight * m_div_vec(deref(LibTag{}, red_loc),k+0);
m_sparse_dimension_idx++;
return lhs;
}, std::vector<double>({-1.000000, 1.000000}));
      }      for(auto const& loc : getEdges(LibTag{}, m_mesh)) {
m_nabla2t2_vec(deref(LibTag{}, loc),k+0) = (m_nabla2t2_vec(deref(LibTag{}, loc),k+0) / m_dual_edge_length(deref(LibTag{}, loc),k+0));
      }      for(auto const& loc : getVertices(LibTag{}, m_mesh)) {
int m_sparse_dimension_idx = 0;
m_rot_vec(deref(LibTag{}, loc),k+0) = reduce(LibTag{}, m_mesh,loc, (::dawn::float_type) 0.0, std::vector<dawn::LocationType>{dawn::LocationType::Vertices, dawn::LocationType::Edges}, [&](auto& lhs, auto red_loc) { lhs += (m_vec(deref(LibTag{}, red_loc),k+0) * m_geofac_rot(deref(LibTag{}, loc),m_sparse_dimension_idx, k+0));
m_sparse_dimension_idx++;
return lhs;
});
      }      for(auto const& loc : getEdges(LibTag{}, m_mesh)) {
int m_sparse_dimension_idx = 0;
m_nabla2t1_vec(deref(LibTag{}, loc),k+0) = reduce(LibTag{}, m_mesh,loc, (::dawn::float_type) 0.0, std::vector<dawn::LocationType>{dawn::LocationType::Edges, dawn::LocationType::Vertices}, [&](auto& lhs, auto red_loc, auto const& weight) {
lhs += weight * m_rot_vec(deref(LibTag{}, red_loc),k+0);
m_sparse_dimension_idx++;
return lhs;
}, std::vector<double>({-1.000000, 1.000000}));
      }      for(auto const& loc : getEdges(LibTag{}, m_mesh)) {
m_nabla2t1_vec(deref(LibTag{}, loc),k+0) = ((m_tangent_orientation(deref(LibTag{}, loc),k+0) * m_nabla2t1_vec(deref(LibTag{}, loc),k+0)) / m_primal_edge_length(deref(LibTag{}, loc),k+0));
      }      for(auto const& loc : getEdges(LibTag{}, m_mesh)) {
m_nabla2_vec(deref(LibTag{}, loc),k+0) = (m_nabla2t2_vec(deref(LibTag{}, loc),k+0) - m_nabla2t1_vec(deref(LibTag{}, loc),k+0));
      }    }}      sync_storages();
    }
  };
  static constexpr const char* s_name = "ICON_laplacian_stencil";
  stencil_64 m_stencil_64;
public:

  ICON_laplacian_stencil(const ICON_laplacian_stencil&) = delete;

  // Members

  ICON_laplacian_stencil(const dawn::mesh_t<LibTag> &mesh, int k_size, dawn::edge_field_t<LibTag, double>& vec, dawn::cell_field_t<LibTag, double>& div_vec, dawn::vertex_field_t<LibTag, double>& rot_vec, dawn::edge_field_t<LibTag, double>& nabla2t1_vec, dawn::edge_field_t<LibTag, double>& nabla2t2_vec, dawn::edge_field_t<LibTag, double>& nabla2_vec, dawn::edge_field_t<LibTag, double>& primal_edge_length, dawn::edge_field_t<LibTag, double>& dual_edge_length, dawn::edge_field_t<LibTag, double>& tangent_orientation, dawn::sparse_vertex_field_t<LibTag, double>& geofac_rot, dawn::sparse_cell_field_t<LibTag, double>& geofac_div) : m_stencil_64(mesh, k_size,vec,div_vec,rot_vec,nabla2t1_vec,nabla2t2_vec,nabla2_vec,primal_edge_length,dual_edge_length,tangent_orientation,geofac_rot,geofac_div){}

  void run() {
    m_stencil_64.run();
;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
