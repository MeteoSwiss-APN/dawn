#define DAWN_GENERATED 1
#define DAWN_BACKEND_T CXXNAIVEICO
#include "interfaces/unstructured_interface.hpp"
namespace dawn_generated {
namespace cxxnaiveico {
template <typename LibTag>
class icon {
private:
  struct stencil_399 {
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
    stencil_399(dawn::mesh_t<LibTag> const& mesh, int k_size,
                dawn::edge_field_t<LibTag, double>& vec,
                dawn::cell_field_t<LibTag, double>& div_vec,
                dawn::vertex_field_t<LibTag, double>& rot_vec,
                dawn::edge_field_t<LibTag, double>& nabla2t1_vec,
                dawn::edge_field_t<LibTag, double>& nabla2t2_vec,
                dawn::edge_field_t<LibTag, double>& nabla2_vec,
                dawn::edge_field_t<LibTag, double>& primal_edge_length,
                dawn::edge_field_t<LibTag, double>& dual_edge_length,
                dawn::edge_field_t<LibTag, double>& tangent_orientation,
                dawn::sparse_vertex_field_t<LibTag, double>& geofac_rot,
                dawn::sparse_cell_field_t<LibTag, double>& geofac_div)
        : m_mesh(mesh), m_k_size(k_size), m_vec(vec), m_div_vec(div_vec), m_rot_vec(rot_vec),
          m_nabla2t1_vec(nabla2t1_vec), m_nabla2t2_vec(nabla2t2_vec), m_nabla2_vec(nabla2_vec),
          m_primal_edge_length(primal_edge_length), m_dual_edge_length(dual_edge_length),
          m_tangent_orientation(tangent_orientation), m_geofac_rot(geofac_rot),
          m_geofac_div(geofac_div) {}

    ~stencil_399() {}

    void sync_storages() {}

    void run() {
      using dawn::deref;
      ;
      {
        for(int k = 0 + 0; k <= (m_k_size == 0 ? 0 : (m_k_size - 1)) + 0 + 0; ++k) {
          for(auto const& loc : getVertices(LibTag{}, m_mesh)) {
            int m_sparse_dimension_idx = 0;
            m_rot_vec(deref(LibTag{}, loc), k + 0) = reduceEdgeToVertex(
                LibTag{}, m_mesh, loc, (::dawn::float_type)0.000000,
                [&](auto& lhs, auto const& red_loc) {
                  lhs += (m_vec(deref(LibTag{}, red_loc), k + 0) *
                          m_geofac_rot(deref(LibTag{}, loc), m_sparse_dimension_idx, k + 0));
                  m_sparse_dimension_idx++;
                  return lhs;
                });
          }
          for(auto const& loc : getCells(LibTag{}, m_mesh)) {
            int m_sparse_dimension_idx = 0;
            m_div_vec(deref(LibTag{}, loc), k + 0) = reduceEdgeToCell(
                LibTag{}, m_mesh, loc, (::dawn::float_type)0.000000,
                [&](auto& lhs, auto const& red_loc) {
                  lhs += (m_vec(deref(LibTag{}, red_loc), k + 0) *
                          m_geofac_div(deref(LibTag{}, loc), m_sparse_dimension_idx, k + 0));
                  m_sparse_dimension_idx++;
                  return lhs;
                });
          }
          for(auto const& loc : getEdges(LibTag{}, m_mesh)) {
            int m_sparse_dimension_idx = 0;
            m_nabla2t1_vec(deref(LibTag{}, loc), k + 0) = reduceVertexToEdge(
                LibTag{}, m_mesh, loc, (::dawn::float_type)0.000000,
                [&](auto& lhs, auto const& red_loc, auto const& weight) {
                  lhs += weight * m_rot_vec(deref(LibTag{}, red_loc), k + 0);
                  m_sparse_dimension_idx++;
                  return lhs;
                },
                std::vector<double>({-1.000000, 1.000000}));
            m_nabla2t1_vec(deref(LibTag{}, loc), k + 0) =
                ((m_tangent_orientation(deref(LibTag{}, loc), k + 0) *
                  m_nabla2t1_vec(deref(LibTag{}, loc), k + 0)) /
                 m_primal_edge_length(deref(LibTag{}, loc), k + 0));
            m_nabla2t2_vec(deref(LibTag{}, loc), k + 0) = reduceCellToEdge(
                LibTag{}, m_mesh, loc, (::dawn::float_type)0.000000,
                [&](auto& lhs, auto const& red_loc, auto const& weight) {
                  lhs += weight * m_div_vec(deref(LibTag{}, red_loc), k + 0);
                  m_sparse_dimension_idx++;
                  return lhs;
                },
                std::vector<double>({-1.000000, 1.000000}));
            m_nabla2t2_vec(deref(LibTag{}, loc), k + 0) =
                (m_nabla2t2_vec(deref(LibTag{}, loc), k + 0) /
                 m_dual_edge_length(deref(LibTag{}, loc), k + 0));
            m_nabla2_vec(deref(LibTag{}, loc), k + 0) =
                (-m_nabla2t1_vec(deref(LibTag{}, loc), k + 0) +
                 m_nabla2t2_vec(deref(LibTag{}, loc), k + 0));
          }
        }
      }
      sync_storages();
    }
  };
  static constexpr const char* s_name = "icon";
  stencil_399 m_stencil_399;

public:
  icon(const icon&) = delete;

  // Members

  icon(const dawn::mesh_t<LibTag>& mesh, int k_size, dawn::edge_field_t<LibTag, double>& vec,
       dawn::cell_field_t<LibTag, double>& div_vec, dawn::vertex_field_t<LibTag, double>& rot_vec,
       dawn::edge_field_t<LibTag, double>& nabla2t1_vec,
       dawn::edge_field_t<LibTag, double>& nabla2t2_vec,
       dawn::edge_field_t<LibTag, double>& nabla2_vec,
       dawn::edge_field_t<LibTag, double>& primal_edge_length,
       dawn::edge_field_t<LibTag, double>& dual_edge_length,
       dawn::edge_field_t<LibTag, double>& tangent_orientation,
       dawn::sparse_vertex_field_t<LibTag, double>& geofac_rot,
       dawn::sparse_cell_field_t<LibTag, double>& geofac_div)
      : m_stencil_399(mesh, k_size, vec, div_vec, rot_vec, nabla2t1_vec, nabla2t2_vec, nabla2_vec,
                      primal_edge_length, dual_edge_length, tangent_orientation, geofac_rot,
                      geofac_div) {}

  void run() {
    m_stencil_399.run();
    ;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
