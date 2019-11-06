#define GRIDTOOLS_CLANG_reference 1
#define GRIDTOOLS_CLANG_BACKEND_T CXXNAIVEICO
#include <gridtools/clang_dsl.hpp>
namespace dawn_generated {
namespace cxxnaiveico {
template <typename LibTag>
class reference {
private:
  struct stencil_107 {
    gtclang::mesh_t<LibTag> const& m_mesh;
    gtclang::cell_field_t<LibTag, double>& m_in_field;
    gtclang::cell_field_t<LibTag, double>& m_out_field;

  public:
    stencil_107(gtclang::mesh_t<LibTag> const& mesh,
                gtclang::cell_field_t<LibTag, double>& in_field,
                gtclang::cell_field_t<LibTag, double>& out_field)
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
          m_out_field(t) =
              (m_in_field(t) + ((gridtools::clang::float_type)0.100000 * m_out_field(t)));
        }
      }
      sync_storages();
    }
  };
  static constexpr const char* s_name = "reference";
  stencil_107 m_stencil_107;

public:
  reference(const reference&) = delete;

  // Members

  reference(const gtclang::mesh_t<LibTag>& mesh, gtclang::cell_field_t<LibTag, double>& in_field,
            gtclang::cell_field_t<LibTag, double>& out_field)
      : m_stencil_107(mesh, in_field, out_field) {}

  void run() {
    m_stencil_107.run();
    ;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
