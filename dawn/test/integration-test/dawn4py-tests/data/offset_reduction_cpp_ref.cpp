#define DAWN_GENERATED 1
#undef DAWN_BACKEND_T
#define DAWN_BACKEND_T CXXNAIVEICO
#define GRIDTOOLS_DAWN_NO_INCLUDE
#include <driver-includes/unstructured_interface.hpp>
#include <driver-includes/unstructured_domain.hpp>
#include <driver-includes/math.hpp>

namespace dawn_generated {
namespace cxxnaiveico {
template <typename LibTag>
class offset_reduction_cpp {
private:
  struct stencil_34 {
    ::dawn::mesh_t<LibTag> const& m_mesh;
    int m_k_size;
    ::dawn::edge_field_t<LibTag, ::dawn::float_type>& m_out_vn_e;
    ::dawn::sparse_edge_field_t<LibTag, ::dawn::float_type>& m_raw_diam_coeff;
    ::dawn::edge_field_t<LibTag, ::dawn::float_type>& m_prism_thick_e;
    ::dawn::sparse_edge_field_t<LibTag, ::dawn::float_type>& m_e2c_aux;
    ::dawn::sparse_edge_field_t<LibTag, ::dawn::float_type>& m_e2c_aux_h;
    ::dawn::unstructured_domain m_unstructured_domain;

  public:
    stencil_34(::dawn::mesh_t<LibTag> const& mesh, int k_size,
               ::dawn::edge_field_t<LibTag, ::dawn::float_type>& out_vn_e,
               ::dawn::sparse_edge_field_t<LibTag, ::dawn::float_type>& raw_diam_coeff,
               ::dawn::edge_field_t<LibTag, ::dawn::float_type>& prism_thick_e,
               ::dawn::sparse_edge_field_t<LibTag, ::dawn::float_type>& e2c_aux,
               ::dawn::sparse_edge_field_t<LibTag, ::dawn::float_type>& e2c_aux_h)
        : m_mesh(mesh), m_k_size(k_size), m_out_vn_e(out_vn_e), m_raw_diam_coeff(raw_diam_coeff),
          m_prism_thick_e(prism_thick_e), m_e2c_aux(e2c_aux), m_e2c_aux_h(e2c_aux_h) {}

    ~stencil_34() {}

    void sync_storages() {}
    static constexpr ::dawn::driver::unstructured_extent out_vn_e_extent = {false, 0, 0};
    static constexpr ::dawn::driver::unstructured_extent raw_diam_coeff_extent = {true, 0, 0};
    static constexpr ::dawn::driver::unstructured_extent prism_thick_e_extent = {true, 0, 0};
    static constexpr ::dawn::driver::unstructured_extent e2c_aux_extent = {true, 0, 0};
    static constexpr ::dawn::driver::unstructured_extent e2c_aux_h_extent = {true, 0, 0};

    void run() {
      using ::dawn::deref;
      {
        for(int k = 0 + 0; k <= (m_k_size == 0 ? 0 : (m_k_size - 1)) + 0 + 0; ++k) {
          for(auto const& loc : getEdges(LibTag{}, m_mesh)) {
            m_out_vn_e(deref(LibTag{}, loc), (k + 0)) = reduce(
                LibTag{}, m_mesh, loc, (::dawn::float_type).0,
                std::vector<::dawn::LocationType>{::dawn::LocationType::Edges,
                                                  ::dawn::LocationType::Cells,
                                                  ::dawn::LocationType::Edges},
                [&, sparse_dimension_idx0 = int(0)](auto& lhs, auto red_loc1,
                                                    auto const& weight) mutable {
                  lhs += weight *
                         (m_raw_diam_coeff(deref(LibTag{}, loc), sparse_dimension_idx0, (k + 0)) *
                          m_prism_thick_e(deref(LibTag{}, red_loc1), (k + 0)));
                  sparse_dimension_idx0++;
                  return lhs;
                },
                std::vector<::dawn::float_type>({m_e2c_aux(deref(LibTag{}, loc), 0, (k + 0)),
                                                 m_e2c_aux(deref(LibTag{}, loc), 0, (k + 0)),
                                                 m_e2c_aux(deref(LibTag{}, loc), 1, (k + 0)),
                                                 m_e2c_aux(deref(LibTag{}, loc), 1, (k + 0))}));
            m_out_vn_e(deref(LibTag{}, loc), (k + 0)) = reduce(
                LibTag{}, m_mesh, loc, (::dawn::float_type).0,
                std::vector<::dawn::LocationType>{::dawn::LocationType::Edges,
                                                  ::dawn::LocationType::Cells,
                                                  ::dawn::LocationType::Edges},
                [&, sparse_dimension_idx0 = int(0)](auto& lhs, auto red_loc1,
                                                    auto const& weight) mutable {
                  lhs += weight *
                         (m_raw_diam_coeff(deref(LibTag{}, loc), sparse_dimension_idx0, (k + 0)) *
                          m_prism_thick_e(deref(LibTag{}, red_loc1), (k + 0)));
                  sparse_dimension_idx0++;
                  return lhs;
                },
                std::vector<::dawn::float_type>(
                    {m_e2c_aux_h(deref(LibTag{}, loc), 0), m_e2c_aux_h(deref(LibTag{}, loc), 0),
                     m_e2c_aux_h(deref(LibTag{}, loc), 1), m_e2c_aux_h(deref(LibTag{}, loc), 1)}));
          }
        }
      }
      sync_storages();
    }
  };
  static constexpr const char* s_name = "offset_reduction_cpp";
  stencil_34 m_stencil_34;

public:
  offset_reduction_cpp(const offset_reduction_cpp&) = delete;

  // Members

  void set_splitter_index(::dawn::LocationType loc, ::dawn::UnstructuredSubdomain subdomain,
                          int offset, int index) {
    m_stencil_34.m_unstructured_domain.set_splitter_index({loc, subdomain, offset}, index);
  }

  offset_reduction_cpp(const ::dawn::mesh_t<LibTag>& mesh, int k_size,
                       ::dawn::edge_field_t<LibTag, ::dawn::float_type>& out_vn_e,
                       ::dawn::sparse_edge_field_t<LibTag, ::dawn::float_type>& raw_diam_coeff,
                       ::dawn::edge_field_t<LibTag, ::dawn::float_type>& prism_thick_e,
                       ::dawn::sparse_edge_field_t<LibTag, ::dawn::float_type>& e2c_aux,
                       ::dawn::sparse_edge_field_t<LibTag, ::dawn::float_type>& e2c_aux_h)
      : m_stencil_34(mesh, k_size, out_vn_e, raw_diam_coeff, prism_thick_e, e2c_aux, e2c_aux_h) {}

  void run() {
    m_stencil_34.run();
    ;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
