#define GRIDTOOLS_CLANG_GENERATED 1
#define GRIDTOOLS_CLANG_BACKEND_T CXXNAIVEICO
#include <driver-includes/interface.hpp>
namespace dawn_generated {
namespace cxxnaiveico {
template <typename LibTag>
class generated {
private:
  struct stencil_107 {
    dawn::mesh_t<LibTag> const& m_mesh;
    dawn::cell_field_t<LibTag, double>& m_in_field;
    dawn::cell_field_t<LibTag, double>& m_out_field;

  public:
    stencil_107(dawn::mesh_t<LibTag> const& mesh, dawn::cell_field_t<LibTag, double>& in_field,
                dawn::cell_field_t<LibTag, double>& out_field)
        : m_mesh(mesh), m_in_field(in_field), m_out_field(out_field) {}

    ~stencil_107() {}

    void sync_storages() {}

    void run() {
      {
        for(auto const& t : getCells(LibTag{}, m_mesh)) {
          int cnt;
          cnt = reduceCellToCell(LibTag{}, m_mesh, t, (int)0,
                                 [&](auto& lhs, auto const& t) { return lhs += (int)1; });
          m_out_field(t) =
              reduceCellToCell(LibTag{}, m_mesh, t, ((-cnt) * m_in_field(t)),
                               [&](auto& lhs, auto const& t) { return lhs += m_in_field(t); });
          m_out_field(t) = (m_in_field(t) + ((dawn::float_type)0.100000 * m_out_field(t)));
        }
      }
      sync_storages();
    }
  };
  static constexpr const char* s_name = "generated";
  stencil_107 m_stencil_107;

public:
  generated(const generated&) = delete;

  // Members

  generated(const dawn::mesh_t<LibTag>& mesh, dawn::cell_field_t<LibTag, double>& in_field,
            dawn::cell_field_t<LibTag, double>& out_field)
      : m_stencil_107(mesh, in_field, out_field) {}

  void run() {
    m_stencil_107.run();
    ;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
