#define DAWN_GENERATED 1
#undef DAWN_BACKEND_T
#define DAWN_BACKEND_T CXXNAIVEICO
#include <driver-includes/unstructured_domain.hpp>
#include <driver-includes/unstructured_interface.hpp>

namespace dawn_generated {
namespace cxxnaiveico {
template <typename LibTag>
class reference_gradient {
private:
  struct stencil_279 {
    ::dawn::mesh_t<LibTag> const& m_mesh;
    int m_k_size;
    ::dawn::cell_field_t<LibTag, ::dawn::float_type>& m_cell_field;
    ::dawn::edge_field_t<LibTag, ::dawn::float_type>& m_edge_field;
    dawn::unstructured_domain m_unstructured_domain;

  public:
    stencil_279(::dawn::mesh_t<LibTag> const& mesh, int k_size,
                ::dawn::cell_field_t<LibTag, ::dawn::float_type>& cell_field,
                ::dawn::edge_field_t<LibTag, ::dawn::float_type>& edge_field)
        : m_mesh(mesh), m_k_size(k_size), m_cell_field(cell_field), m_edge_field(edge_field) {}

    ~stencil_279() {}

    void sync_storages() {}
    static constexpr ::dawn::driver::unstructured_extent cell_field_extent = {true, 0, 0};
    static constexpr ::dawn::driver::unstructured_extent edge_field_extent = {true, 0, 0};

    void run() {
      using ::dawn::deref;
      {
        for(int k = 0 + 0; k <= (m_k_size == 0 ? 0 : (m_k_size - 1)) + 0 + 0; ++k) {
          for(auto const& loc : getEdges(LibTag{}, m_mesh)) {
            {
              int sparse_dimension_idx0 = 0;
              m_edge_field(deref(LibTag{}, loc), k + 0) = reduce(
                  LibTag{}, m_mesh, loc, (::dawn::float_type)0.000000,
                  std::vector<::dawn::LocationType>{::dawn::LocationType::Edges,
                                                    ::dawn::LocationType::Cells},
                  [&](auto& lhs, auto red_loc1, auto const& weight) {
                    lhs += weight * m_cell_field(deref(LibTag{}, red_loc1), k + 0);
                    sparse_dimension_idx0++;
                    return lhs;
                  },
                  std::vector<::dawn::float_type>(
                      {(::dawn::float_type)1.000000, (::dawn::float_type)-1.000000}));
            }
          }
          for(auto const& loc : getCells(LibTag{}, m_mesh)) {
            {
              int sparse_dimension_idx0 = 0;
              m_cell_field(deref(LibTag{}, loc), k + 0) = reduce(
                  LibTag{}, m_mesh, loc, (::dawn::float_type)0.000000,
                  std::vector<::dawn::LocationType>{::dawn::LocationType::Cells,
                                                    ::dawn::LocationType::Edges},
                  [&](auto& lhs, auto red_loc1, auto const& weight) {
                    lhs += weight * m_edge_field(deref(LibTag{}, red_loc1), k + 0);
                    sparse_dimension_idx0++;
                    return lhs;
                  },
                  std::vector<::dawn::float_type>(
                      {(::dawn::float_type)0.500000, (::dawn::float_type)0.000000,
                       (::dawn::float_type)0.000000, (::dawn::float_type)0.500000}));
            }
          }
        }
      }
      sync_storages();
    }
  };
  static constexpr const char* s_name = "gradient";
  stencil_279 m_stencil_279;

public:
  reference_gradient(const reference_gradient&) = delete;

  // Members

  void set_splitter_index(::dawn::LocationType loc, dawn::UnstructuredIterationSpace space,
                          int offset, int index) {
    m_stencil_279.m_unstructured_domain.set_splitter_index({loc, space, offset}, index);
  }

  reference_gradient(const ::dawn::mesh_t<LibTag>& mesh, int k_size,
                     ::dawn::cell_field_t<LibTag, ::dawn::float_type>& cell_field,
                     ::dawn::edge_field_t<LibTag, ::dawn::float_type>& edge_field)
      : m_stencil_279(mesh, k_size, cell_field, edge_field) {}

  void run() {
    m_stencil_279.run();
    ;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
