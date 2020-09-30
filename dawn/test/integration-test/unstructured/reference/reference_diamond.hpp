#define DAWN_GENERATED 1
#undef DAWN_BACKEND_T
#define DAWN_BACKEND_T CXXNAIVEICO
#include <driver-includes/unstructured_domain.hpp>
#include <driver-includes/unstructured_interface.hpp>

namespace dawn_generated {
namespace cxxnaiveico {
template <typename LibTag>
class reference_diamond {
private:
  struct stencil_300 {
    ::dawn::mesh_t<LibTag> const& m_mesh;
    int m_k_size;
    ::dawn::edge_field_t<LibTag, ::dawn::float_type>& m_edge_field;
    ::dawn::vertex_field_t<LibTag, ::dawn::float_type>& m_vertex_field;
    dawn::unstructured_domain m_unstructured_domain;

  public:
    stencil_300(::dawn::mesh_t<LibTag> const& mesh, int k_size,
                ::dawn::edge_field_t<LibTag, ::dawn::float_type>& edge_field,
                ::dawn::vertex_field_t<LibTag, ::dawn::float_type>& vertex_field)
        : m_mesh(mesh), m_k_size(k_size), m_edge_field(edge_field), m_vertex_field(vertex_field) {}

    ~stencil_300() {}

    void sync_storages() {}
    static constexpr ::dawn::driver::unstructured_extent edge_field_extent = {false, 0, 0};
    static constexpr ::dawn::driver::unstructured_extent vertex_field_extent = {true, 0, 0};

    void run() {
      using ::dawn::deref;
      {
        for(int k = 0 + 0; k <= (m_k_size == 0 ? 0 : (m_k_size - 1)) + 0 + 0; ++k) {
          for(auto const& loc : getEdges(LibTag{}, m_mesh)) {
            {
              int sparse_dimension_idx0 = 0;
              m_edge_field(deref(LibTag{}, loc), k + 0) =
                  reduce(LibTag{}, m_mesh, loc, (::dawn::float_type)0.000000,
                         std::vector<::dawn::LocationType>{::dawn::LocationType::Edges,
                                                           ::dawn::LocationType::Cells,
                                                           ::dawn::LocationType::Vertices},
                         [&](auto& lhs, auto red_loc1) {
                           lhs += m_vertex_field(deref(LibTag{}, red_loc1), k + 0);
                           sparse_dimension_idx0++;
                           return lhs;
                         });
            }
          }
        }
      }
      sync_storages();
    }
  };
  static constexpr const char* s_name = "diamond";
  stencil_300 m_stencil_300;

public:
  reference_diamond(const reference_diamond&) = delete;

  // Members

  void set_splitter_index(::dawn::LocationType loc, dawn::UnstructuredIterationSpace space,
                          int offset, int index) {
    m_stencil_300.m_unstructured_domain.set_splitter_index({loc, space, offset}, index);
  }

  reference_diamond(const ::dawn::mesh_t<LibTag>& mesh, int k_size,
                    ::dawn::edge_field_t<LibTag, ::dawn::float_type>& edge_field,
                    ::dawn::vertex_field_t<LibTag, ::dawn::float_type>& vertex_field)
      : m_stencil_300(mesh, k_size, edge_field, vertex_field) {}

  void run() {
    m_stencil_300.run();
    ;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
