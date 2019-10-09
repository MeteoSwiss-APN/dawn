namespace dawn_generated {
namespace cxxnaiveico {

class unstructured_stencil {
private:
  struct stencil_13 {
    Mesh const& m_mesh;
    Field<double>& m_in;
    Field<double>& m_out;

  public:
    stencil_13(Mesh const& mesh, Field<double>& in, Field<double>& out)
        : m_mesh(mesh), m_in(in), m_out(out) {}

    ~stencil_13() {}

    void sync_storages() {}

    void run() {
      {
        for(auto const& t : getTriangles(m_mesh)) {
          m_out[t] =
              reduce(cellNeighboursOfCell(m_mesh, t), m_in[t], [&](auto& lhs, auto const& t) {
                return lhs += (gridtools::clang::float_type)1.0;
              });
        }
      }
      sync_storages();
    }
  };
  static constexpr const char* s_name = "unstructured_stencil";
  stencil_13 m_stencil_13;

public:
  unstructured_stencil(const unstructured_stencil&) = delete;

  // Members

  unstructured_stencil(const Mesh& mesh, Field<double>& in, Field<double>& out)
      : m_stencil_13(mesh, in, out) {}

  void run() {
    m_stencil_13.run();
    ;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
