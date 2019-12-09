#define GRIDTOOLS_CLANG_GENERATED 1
#define GRIDTOOLS_CLANG_BACKEND_T CXXNAIVEICO
#include <driver-includes/unstructured_interface.hpp>
namespace dawn_generated {
namespace cxxnaiveico {
template <typename LibTag>
class gradient {
private:
  struct stencil_165 {
    dawn::mesh_t<LibTag> const& m_mesh;
    int m_k_size;
    dawn::cell_field_t<LibTag, double>& m_cell_field;
    dawn::cell_field_t<LibTag, double>& m_edge_field;

  public:
    stencil_165(dawn::mesh_t<LibTag> const& mesh, int k_size,
                dawn::cell_field_t<LibTag, double>& cell_field,
                dawn::cell_field_t<LibTag, double>& edge_field)
        : m_mesh(mesh), m_k_size(k_size), m_cell_field(cell_field), m_edge_field(edge_field) {}

    ~stencil_165() {}

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
                                 std::vector<float>({1.000000, 1.000000}));
          }
          for(auto const& loc : getCells(LibTag{}, m_mesh)) {
            m_cell_field(deref(LibTag{}, loc), k + 0) =
                reduceEdgeToCell(LibTag{}, m_mesh, loc, (::dawn::float_type)0.000000,
                                 [&](auto& lhs, auto const& red_loc, auto const& weight) {
                                   return lhs +=
                                          weight * m_edge_field(deref(LibTag{}, red_loc), k + 0);
                                 },
                                 std::vector<float>({0.250000, 0.250000, 0.250000, 0.250000}));
          }
        }
      }
      sync_storages();
    }
  };
  static constexpr const char* s_name = "gradient";
  stencil_165 m_stencil_165;

public:
  gradient(const gradient&) = delete;

  // Members

  gradient(const dawn::mesh_t<LibTag>& mesh, int k_size,
           dawn::cell_field_t<LibTag, double>& cell_field,
           dawn::cell_field_t<LibTag, double>& edge_field)
      : m_stencil_165(mesh, k_size, cell_field, edge_field) {}

  void run() {
    m_stencil_165.run();
    ;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
