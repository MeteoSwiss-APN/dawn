namespace dawn_generated{
namespace cxxnaiveico{
template<typename LibTag>
class unstructured_stencil {
private:

  struct stencil_13 {
    gtclang::mesh_t<LibTag> const& m_mesh;
    int m_k_size;
    gtclang::cell_field_t<LibTag, double>& m_in;
    gtclang::cell_field_t<LibTag, double>& m_out;
  public:

    stencil_13(gtclang::mesh_t<LibTag> const &mesh, int k_size, gtclang::cell_field_t<LibTag, double>&in, gtclang::cell_field_t<LibTag, double>&out) : m_mesh(mesh), m_k_size(k_size), m_in(in), m_out(out){}

    ~stencil_13() {
    }

    void sync_storages() {
    }

    void run() {
      using gtclang::deref;;
{
    for(int k = 0+0; k <= ( m_k_size == 0 ? 0 : (m_k_size - 1)) + 0+0; ++k) {
      for(auto const& loc : getCells(LibTag{}, m_mesh)) {
m_out(deref(LibTag{}, loc),k+0) = reduceCellToCell(LibTag{}, m_mesh, loc, m_in(deref(LibTag{}, loc),k+0), [&](auto& lhs, auto const& red_loc) { return lhs += (gridtools::clang::float_type) 1.0;});
      }    }}      sync_storages();
    }
  };
  static constexpr const char* s_name = "unstructured_stencil";
  stencil_13 m_stencil_13;
public:

  unstructured_stencil(const unstructured_stencil&) = delete;

  // Members

  unstructured_stencil(const gtclang::mesh_t<LibTag> &mesh, int k_size, gtclang::cell_field_t<LibTag, double>& in, gtclang::cell_field_t<LibTag, double>& out) : m_stencil_13(mesh, k_size,in,out){}

  void run() {
    m_stencil_13.run();
;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
