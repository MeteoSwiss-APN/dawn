#define DAWN_GENERATED 1
#define DAWN_BACKEND_T CXXNAIVEICO
#include <driver-includes/unstructured_interface.hpp>
namespace dawn_generated {
namespace cxxnaiveico {
template <typename LibTag>
class reference_gradient {
private:
  struct stencil_284 {
    dawn::mesh_t<LibTag> const& m_mesh;
    int m_k_size;
    dawn::cell_field_t<LibTag, double>& m_cell_field;
    dawn::edge_field_t<LibTag, double>& m_edge_field;

  public:
    stencil_284(dawn::mesh_t<LibTag> const& mesh, int k_size,
                dawn::cell_field_t<LibTag, double>& cell_field,
                dawn::edge_field_t<LibTag, double>& edge_field)
        : m_mesh(mesh), m_k_size(k_size), m_cell_field(cell_field), m_edge_field(edge_field) {}

    ~stencil_284() {}

    void sync_storages() {}

    void run() {
      using dawn::deref;
      ;
      {
        for(int k = 0 + 0; k <= (m_k_size == 0 ? 0 : (m_k_size - 1)) + 0 + 0; ++k) {
          for(auto const& loc : getEdges(LibTag{}, m_mesh)) {
            m_edge_field(deref(LibTag{}, loc), k + 0) =
                reduceCellToEdge(LibTag{}, m_mesh, loc, (::dawn::float_type)0.000000,
                                 [&](auto& lhs, auto const& red_loc, auto const& weight) {
                                   return lhs +=
                                          weight * m_cell_field(deref(LibTag{}, red_loc), k + 0);
                                 },
                                 std::vector<float>({1.000000, -1.000000}));
          }
          for(auto const& loc : getCells(LibTag{}, m_mesh)) {
            m_cell_field(deref(LibTag{}, loc), k + 0) =
                reduceEdgeToCell(LibTag{}, m_mesh, loc, (::dawn::float_type)0.000000,
                                 [&](auto& lhs, auto const& red_loc, auto const& weight) {
                                   return lhs +=
                                          weight * m_edge_field(deref(LibTag{}, red_loc), k + 0);
                                 },
                                 std::vector<float>({0.500000, 0.000000, 0.000000, 0.500000}));
          }
        }
      }
      sync_storages();
    }
  };
  static constexpr const char* s_name = "gradient";
  stencil_284 m_stencil_284;

public:
  reference_gradient(const reference_gradient&) = delete;

  // Members

  reference_gradient(const dawn::mesh_t<LibTag>& mesh, int k_size,
                     dawn::cell_field_t<LibTag, double>& cell_field,
                     dawn::edge_field_t<LibTag, double>& edge_field)
      : m_stencil_284(mesh, k_size, cell_field, edge_field) {}

  void run() {
    m_stencil_284.run();
    ;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
