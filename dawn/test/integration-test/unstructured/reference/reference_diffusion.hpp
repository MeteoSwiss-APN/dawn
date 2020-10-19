#define DAWN_GENERATED 1
#undef DAWN_BACKEND_T
#define DAWN_BACKEND_T CXXNAIVEICO
#include <driver-includes/unstructured_domain.hpp>
#include <driver-includes/unstructured_interface.hpp>

namespace dawn_generated {
namespace cxxnaiveico {
template <typename LibTag>
class reference_diffusion {
private:
  struct stencil_237 {
    ::dawn::mesh_t<LibTag> const& m_mesh;
    int m_k_size;
    ::dawn::cell_field_t<LibTag, ::dawn::float_type>& m_in_field;
    ::dawn::cell_field_t<LibTag, ::dawn::float_type>& m_out_field;
    dawn::unstructured_domain m_unstructured_domain;

  public:
    stencil_237(::dawn::mesh_t<LibTag> const& mesh, int k_size,
                ::dawn::cell_field_t<LibTag, ::dawn::float_type>& in_field,
                ::dawn::cell_field_t<LibTag, ::dawn::float_type>& out_field)
        : m_mesh(mesh), m_k_size(k_size), m_in_field(in_field), m_out_field(out_field) {}

    ~stencil_237() {}

    void sync_storages() {}
    static constexpr ::dawn::driver::unstructured_extent in_field_extent = {true, 0, 0};
    static constexpr ::dawn::driver::unstructured_extent out_field_extent = {false, 0, 0};

    void run() {
      using ::dawn::deref;
      {
        for(int k = 0 + 0; k <= (m_k_size == 0 ? 0 : (m_k_size - 1)) + 0 + 0; ++k) {
          for(auto const& loc : getCells(LibTag{}, m_mesh)) {
            int cnt;
            {
              int sparse_dimension_idx0 = 0;
              cnt = reduce(LibTag{}, m_mesh, loc, (int)0,
                           std::vector<::dawn::LocationType>{::dawn::LocationType::Cells,
                                                             ::dawn::LocationType::Edges,
                                                             ::dawn::LocationType::Cells},
                           [&](auto& lhs, auto red_loc1) {
                             lhs += (int)1;
                             sparse_dimension_idx0++;
                             return lhs;
                           });
            }
            {
              int sparse_dimension_idx0 = 0;
              m_out_field(deref(LibTag{}, loc), k + 0) =
                  reduce(LibTag{}, m_mesh, loc, ((-cnt) * m_in_field(deref(LibTag{}, loc), k + 0)),
                         std::vector<::dawn::LocationType>{::dawn::LocationType::Cells,
                                                           ::dawn::LocationType::Edges,
                                                           ::dawn::LocationType::Cells},
                         [&](auto& lhs, auto red_loc1) {
                           lhs += m_in_field(deref(LibTag{}, red_loc1), k + 0);
                           sparse_dimension_idx0++;
                           return lhs;
                         });
            }
            m_out_field(deref(LibTag{}, loc), k + 0) =
                (m_in_field(deref(LibTag{}, loc), k + 0) +
                 ((::dawn::float_type)0.100000 * m_out_field(deref(LibTag{}, loc), k + 0)));
          }
        }
      }
      sync_storages();
    }
  };
  static constexpr const char* s_name = "diffusion";
  stencil_237 m_stencil_237;

public:
  reference_diffusion(const reference_diffusion&) = delete;

  // Members

  void set_splitter_index(::dawn::LocationType loc, dawn::UnstructuredIterationSpace space,
                          int offset, int index) {
    m_stencil_237.m_unstructured_domain.set_splitter_index({loc, space, offset}, index);
  }

  reference_diffusion(const ::dawn::mesh_t<LibTag>& mesh, int k_size,
                      ::dawn::cell_field_t<LibTag, ::dawn::float_type>& in_field,
                      ::dawn::cell_field_t<LibTag, ::dawn::float_type>& out_field)
      : m_stencil_237(mesh, k_size, in_field, out_field) {}

  void run() {
    m_stencil_237.run();
    ;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
