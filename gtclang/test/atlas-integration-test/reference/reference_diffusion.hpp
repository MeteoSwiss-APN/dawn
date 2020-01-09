#define DAWN_GENERATED 1
#define DAWN_BACKEND_T CXXNAIVEICO
#include <driver-includes/unstructured_interface.hpp>
namespace dawn_generated {
namespace cxxnaiveico {
template <typename LibTag>
class reference_diffusion {
private:
  struct stencil_129 {
    dawn::mesh_t<LibTag> const& m_mesh;
    int m_k_size;
    dawn::cell_field_t<LibTag, double>& m_in_field;
    dawn::cell_field_t<LibTag, double>& m_out_field;

  public:
    stencil_129(dawn::mesh_t<LibTag> const& mesh, int k_size,
                dawn::cell_field_t<LibTag, double>& in_field,
                dawn::cell_field_t<LibTag, double>& out_field)
        : m_mesh(mesh), m_k_size(k_size), m_in_field(in_field), m_out_field(out_field) {}

    ~stencil_129() {}

    void sync_storages() {}

    void run() {
      using dawn::deref;
      ;
      {
        for(int k = 0 + 0; k <= (m_k_size == 0 ? 0 : (m_k_size - 1)) + 0 + 0; ++k) {
          for(auto const& loc : getCells(LibTag{}, m_mesh)) {
            int cnt;
            cnt = reduceCellToCell(LibTag{}, m_mesh, loc, (int)0,
                                   [&](auto& lhs, auto const& red_loc) { return lhs += (int)1; });
            m_out_field(deref(LibTag{}, loc), k + 0) = reduceCellToCell(
                LibTag{}, m_mesh, loc, ((-cnt) * m_in_field(deref(LibTag{}, loc), k + 0)),
                [&](auto& lhs, auto const& red_loc) {
                  return lhs += m_in_field(deref(LibTag{}, red_loc), k + 0);
                });
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
  stencil_129 m_stencil_129;

public:
  reference_diffusion(const reference_diffusion&) = delete;

  // Members

  reference_diffusion(const dawn::mesh_t<LibTag>& mesh, int k_size,
                      dawn::cell_field_t<LibTag, double>& in_field,
                      dawn::cell_field_t<LibTag, double>& out_field)
      : m_stencil_129(mesh, k_size, in_field, out_field) {}

  void run() {
    m_stencil_129.run();
    ;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
