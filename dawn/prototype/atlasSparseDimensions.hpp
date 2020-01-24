#define DAWN_GENERATED 1
#define DAWN_BACKEND_T CXXNAIVEICO
#include <driver-includes/unstructured_interface.hpp>
namespace dawn_generated {
namespace cxxnaiveico {
template <typename LibTag>
class sparseDimension {
private:
  struct stencil_228 {
    dawn::mesh_t<LibTag> const& m_mesh;
    int m_k_size;
    dawn::cell_field_t<LibTag, double>& m_cell_field;
    dawn::edge_field_t<LibTag, double>& m_edge_field;
    dawn::sparse_dimension_t<LibTag, double>& m_sparse_dimension;

  public:
    stencil_228(dawn::mesh_t<LibTag> const& mesh, int k_size,
                dawn::cell_field_t<LibTag, double>& cell_field,
                dawn::edge_field_t<LibTag, double>& edge_field,
                dawn::sparse_dimension_t<LibTag, double>& sparse_dimension)
        : m_mesh(mesh), m_k_size(k_size), m_cell_field(cell_field), m_edge_field(edge_field),
          m_sparse_dimension(sparse_dimension) {}

    ~stencil_228() {}

    void sync_storages() {}

    void run() {
      using dawn::deref;
      ;
      {
        for(int k = 0 + 0; k <= (m_k_size == 0 ? 0 : (m_k_size - 1)) + 0 + 0; ++k) {
          for(auto const& loc : getCells(LibTag{}, m_mesh)) {
            m_cell_field(deref(LibTag{}, loc), k + 0) = reduceEdgeToCell(
                LibTag{}, m_mesh, loc, k, (::dawn::float_type)0.000000,
                [&](auto& lhs, auto const& red_loc, auto const& weight,
                    auto const& sparse_dimension) {
                  return lhs +=
                         weight * sparse_dimension * m_edge_field(deref(LibTag{}, red_loc), k + 0);
                },
                std::vector<double>({1.000000, 1.000000, 1.000000, 1.00000}), m_sparse_dimension);
          }
        }
      }
      sync_storages();
    }
  };
  static constexpr const char* s_name = "sparseDimension";
  stencil_228 m_stencil_228;

public:
  sparseDimension(const sparseDimension&) = delete;

  // Members
  sparseDimension(const dawn::mesh_t<LibTag>& mesh, int k_size,
                  dawn::cell_field_t<LibTag, double>& cell_field,
                  dawn::edge_field_t<LibTag, double>& edge_field,
                  dawn::sparse_dimension_t<LibTag, double>& sparse_dimension)
      : m_stencil_228(mesh, k_size, cell_field, edge_field, sparse_dimension) {}

  void run() {
    m_stencil_228.run();
    ;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
