#define GRIDTOOLS_CLANG_GENERATED 1
#define GRIDTOOLS_CLANG_BACKEND_T CXXNAIVEICO
#include <gridtools/clang_dsl.hpp>
namespace dawn_generated {
namespace cxxnaiveico {
template <typename LibTag>
class generated {
private:
  struct stencil_127 {
    gtclang::mesh_t<LibTag> const& m_mesh;
    int m_k_size;
    gtclang::cell_field_t<LibTag, double>& m_in_field;
    gtclang::cell_field_t<LibTag, double>& m_out_field;

  public:
    stencil_127(gtclang::mesh_t<LibTag> const& mesh, int k_size,
                gtclang::cell_field_t<LibTag, double>& in_field,
                gtclang::cell_field_t<LibTag, double>& out_field)
        : m_mesh(mesh), m_k_size(k_size), m_in_field(in_field), m_out_field(out_field) {}

    ~stencil_127() {}

    void sync_storages() {}

    void run() {
      {
        for(int k = 0 + 0; k <= (m_k_size == 0 ? 0 : (m_k_size - 1)) + 0 + 0; ++k) {
          for(auto const& t : getCells(LibTag{}, m_mesh)) {
            int cnt;
            cnt = reduceCellToCell(LibTag{}, m_mesh, t, (int)0,
                                   [&](auto& lhs, auto const& t) { return lhs += (int)1; });
            m_out_field(t, k + 0) = reduceCellToCell(
                LibTag{}, m_mesh, t, ((-cnt) * m_in_field(t, k + 0)),
                [&](auto& lhs, auto const& t) { return lhs += m_in_field(t, k + 0); });
            m_out_field(t, k + 0) =
                (m_in_field(t, k + 0) +
                 ((gridtools::clang::float_type)0.100000 * m_out_field(t, k + 0)));
          }
        }
      }
      sync_storages();
    }
  };
  static constexpr const char* s_name = "generated";
  stencil_127 m_stencil_127;

public:
  generated(const generated&) = delete;

  // Members

  generated(const gtclang::mesh_t<LibTag>& mesh, int k_size,
            gtclang::cell_field_t<LibTag, double>& in_field,
            gtclang::cell_field_t<LibTag, double>& out_field)
      : m_stencil_127(mesh, k_size, in_field, out_field) {}

  void run() {
    m_stencil_127.run();
    ;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
