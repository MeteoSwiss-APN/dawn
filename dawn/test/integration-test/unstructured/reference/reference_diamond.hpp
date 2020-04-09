#define DAWN_GENERATED 1
#undef DAWN_BACKEND_T
#define DAWN_BACKEND_T CXXNAIVEICO
#include <driver-includes/unstructured_interface.hpp>
namespace dawn_generated {
namespace cxxnaiveico {
template <typename LibTag>
class reference_diamond {
private:
  struct stencil_329 {
    dawn::mesh_t<LibTag> const& m_mesh;
    int m_k_size;
    dawn::edge_field_t<LibTag, double>& m_edge_field;
    dawn::vertex_field_t<LibTag, double>& m_vertex_field;

  public:
    stencil_329(dawn::mesh_t<LibTag> const& mesh, int k_size,
                dawn::edge_field_t<LibTag, double>& edge_field,
                dawn::vertex_field_t<LibTag, double>& vertex_field)
        : m_mesh(mesh), m_k_size(k_size), m_edge_field(edge_field), m_vertex_field(vertex_field) {}

    ~stencil_329() {}

    void sync_storages() {}
    static constexpr dawn::driver::unstructured_extent edge_field_extent = {false, 0, 0};
    static constexpr dawn::driver::unstructured_extent vertex_field_extent = {true, 0, 0};

    void run() {
      using dawn::deref;
      {
        for(int k = 0 + 0; k <= (m_k_size == 0 ? 0 : (m_k_size - 1)) + 0 + 0; ++k) {
          for(auto const& loc : getEdges(LibTag{}, m_mesh)) {
            int m_sparse_dimension_idx = 0;
            m_edge_field(deref(LibTag{}, loc), k + 0) =
                reduce(LibTag{}, m_mesh, loc, (::dawn::float_type)0.000000,
                       std::vector<dawn::LocationType>{dawn::LocationType::Edges,
                                                       dawn::LocationType::Cells,
                                                       dawn::LocationType::Vertices},
                       [&](auto& lhs, auto red_loc) {
                         lhs += m_vertex_field(deref(LibTag{}, red_loc), k + 0);
                         m_sparse_dimension_idx++;
                         return lhs;
                       });
          }
        }
      }
      sync_storages();
    }
  };
  static constexpr const char* s_name = "diamond";
  stencil_329 m_stencil_329;

public:
  reference_diamond(const reference_diamond&) = delete;

  // Members

  reference_diamond(const dawn::mesh_t<LibTag>& mesh, int k_size,
                    dawn::edge_field_t<LibTag, double>& edge_field,
                    dawn::vertex_field_t<LibTag, double>& vertex_field)
      : m_stencil_329(mesh, k_size, edge_field, vertex_field) {}

  void run() {
    m_stencil_329.run();
    ;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
