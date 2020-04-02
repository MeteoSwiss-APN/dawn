#define DAWN_GENERATED 1
#undef DAWN_BACKEND_T
#define DAWN_BACKEND_T CXXNAIVEICO
#include <driver-includes/unstructured_interface.hpp>
namespace dawn_generated {
namespace cxxnaiveico {
template <typename LibTag>
class reference_diamondWeights {
private:
  struct stencil_388 {
    dawn::mesh_t<LibTag> const& m_mesh;
    int m_k_size;
    dawn::edge_field_t<LibTag, double>& m_out;
    dawn::edge_field_t<LibTag, double>& m_inv_edge_length;
    dawn::edge_field_t<LibTag, double>& m_inv_vert_length;
    dawn::vertex_field_t<LibTag, double>& m_in;

  public:
    stencil_388(dawn::mesh_t<LibTag> const& mesh, int k_size,
                dawn::edge_field_t<LibTag, double>& out,
                dawn::edge_field_t<LibTag, double>& inv_edge_length,
                dawn::edge_field_t<LibTag, double>& inv_vert_length,
                dawn::vertex_field_t<LibTag, double>& in)
        : m_mesh(mesh), m_k_size(k_size), m_out(out), m_inv_edge_length(inv_edge_length),
          m_inv_vert_length(inv_vert_length), m_in(in) {}

    ~stencil_388() {}

    void sync_storages() {}
    static constexpr dawn::driver::unstructured_extent out_extent = {false, 0, 0};
    static constexpr dawn::driver::unstructured_extent inv_edge_length_extent = {false, 0, 0};
    static constexpr dawn::driver::unstructured_extent inv_vert_length_extent = {false, 0, 0};
    static constexpr dawn::driver::unstructured_extent in_extent = {true, 0, 0};

    void run() {
      using dawn::deref;
      {
        for(int k = 0 + 0; k <= (m_k_size == 0 ? 0 : (m_k_size - 1)) + 0 + 0; ++k) {
          for(auto const& loc : getEdges(LibTag{}, m_mesh)) {
            int m_sparse_dimension_idx = 0;
            m_out(deref(LibTag{}, loc), k + 0) = reduce(
                LibTag{}, m_mesh, loc, (::dawn::float_type)0.000000,
                std::vector<dawn::LocationType>{dawn::LocationType::Edges,
                                                dawn::LocationType::Cells,
                                                dawn::LocationType::Vertices},
                [&](auto& lhs, auto red_loc, auto const& weight) {
                  lhs += weight * m_in(deref(LibTag{}, red_loc), k + 0);
                  m_sparse_dimension_idx++;
                  return lhs;
                },
                std::vector<::dawn::float_type>(
                    {(m_inv_edge_length(deref(LibTag{}, loc), k + 0) *
                      m_inv_edge_length(deref(LibTag{}, loc), k + 0)),
                     (m_inv_edge_length(deref(LibTag{}, loc), k + 0) *
                      m_inv_edge_length(deref(LibTag{}, loc), k + 0)),
                     (m_inv_vert_length(deref(LibTag{}, loc), k + 0) *
                      m_inv_vert_length(deref(LibTag{}, loc), k + 0)),
                     (m_inv_vert_length(deref(LibTag{}, loc), k + 0) *
                      m_inv_vert_length(deref(LibTag{}, loc), k + 0))}));
          }
        }
      }
      sync_storages();
    }
  };
  static constexpr const char* s_name = "diamond";
  stencil_388 m_stencil_388;

public:
  reference_diamondWeights(const reference_diamondWeights&) = delete;

  // Members

  reference_diamondWeights(const dawn::mesh_t<LibTag>& mesh, int k_size,
                           dawn::edge_field_t<LibTag, double>& out,
                           dawn::edge_field_t<LibTag, double>& inv_edge_length,
                           dawn::edge_field_t<LibTag, double>& inv_vert_length,
                           dawn::vertex_field_t<LibTag, double>& in)
      : m_stencil_388(mesh, k_size, out, inv_edge_length, inv_vert_length, in) {}

  void run() {
    m_stencil_388.run();
    ;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
