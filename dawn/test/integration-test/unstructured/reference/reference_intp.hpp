#define DAWN_GENERATED 1
#undef DAWN_BACKEND_T
#define DAWN_BACKEND_T CXXNAIVEICO
#include <driver-includes/unstructured_domain.hpp>
#include <driver-includes/unstructured_interface.hpp>

namespace dawn_generated {
namespace cxxnaiveico {
template <typename LibTag>
class reference_intp {
private:
  struct stencil_368 {
    ::dawn::mesh_t<LibTag> const& m_mesh;
    int m_k_size;
    ::dawn::cell_field_t<LibTag, ::dawn::float_type>& m_in;
    ::dawn::cell_field_t<LibTag, ::dawn::float_type>& m_out;
    dawn::unstructured_domain m_unstructured_domain;

  public:
    stencil_368(::dawn::mesh_t<LibTag> const& mesh, int k_size,
                ::dawn::cell_field_t<LibTag, ::dawn::float_type>& in,
                ::dawn::cell_field_t<LibTag, ::dawn::float_type>& out)
        : m_mesh(mesh), m_k_size(k_size), m_in(in), m_out(out) {}

    ~stencil_368() {}

    void sync_storages() {}
    static constexpr ::dawn::driver::unstructured_extent in_extent = {true, 0, 0};
    static constexpr ::dawn::driver::unstructured_extent out_extent = {false, 0, 0};

    void run() {
      using ::dawn::deref;
      {
        for(int k = 0 + 0; k <= (m_k_size == 0 ? 0 : (m_k_size - 1)) + 0 + 0; ++k) {
          for(auto const& loc : getCells(LibTag{}, m_mesh)) {
            {
              int sparse_dimension_idx0 = 0;
              m_out(deref(LibTag{}, loc), k + 0) =
                  reduce(LibTag{}, m_mesh, loc, (::dawn::float_type)0.000000,
                         std::vector<::dawn::LocationType>{
                             ::dawn::LocationType::Cells, ::dawn::LocationType::Edges,
                             ::dawn::LocationType::Cells, ::dawn::LocationType::Edges,
                             ::dawn::LocationType::Cells},
                         [&](auto& lhs, auto red_loc1) {
                           lhs += m_in(deref(LibTag{}, red_loc1), k + 0);
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
  static constexpr const char* s_name = "intp";
  stencil_368 m_stencil_368;

public:
  reference_intp(const reference_intp&) = delete;

  // Members

  void set_splitter_index(::dawn::LocationType loc, dawn::UnstructuredIterationSpace space,
                          int offset, int index) {
    m_stencil_368.m_unstructured_domain.set_splitter_index({loc, space, offset}, index);
  }

  reference_intp(const ::dawn::mesh_t<LibTag>& mesh, int k_size,
                 ::dawn::cell_field_t<LibTag, ::dawn::float_type>& in,
                 ::dawn::cell_field_t<LibTag, ::dawn::float_type>& out)
      : m_stencil_368(mesh, k_size, in, out) {}

  void run() {
    m_stencil_368.run();
    ;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
