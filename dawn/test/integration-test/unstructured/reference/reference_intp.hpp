#define DAWN_GENERATED 1
#undef DAWN_BACKEND_T
#define DAWN_BACKEND_T CXXNAIVEICO
#include <driver-includes/unstructured_interface.hpp>
namespace dawn_generated {
namespace cxxnaiveico {
template <typename LibTag>
class reference_intp {
private:
  struct stencil_350 {
    dawn::mesh_t<LibTag> const& m_mesh;
    int m_k_size;
    dawn::cell_field_t<LibTag, double>& m_in;
    dawn::cell_field_t<LibTag, double>& m_out;

  public:
    stencil_350(dawn::mesh_t<LibTag> const& mesh, int k_size,
                dawn::cell_field_t<LibTag, double>& in, dawn::cell_field_t<LibTag, double>& out)
        : m_mesh(mesh), m_k_size(k_size), m_in(in), m_out(out) {}

    ~stencil_350() {}

    void sync_storages() {}
    static constexpr dawn::driver::unstructured_extent in_extent = {true, 0, 0};
    static constexpr dawn::driver::unstructured_extent out_extent = {false, 0, 0};

    void run() {
      using dawn::deref;
      {
        for(int k = 0 + 0; k <= (m_k_size == 0 ? 0 : (m_k_size - 1)) + 0 + 0; ++k) {
          for(auto const& loc : getCells(LibTag{}, m_mesh)) {
            int m_sparse_dimension_idx = 0;
            m_out(deref(LibTag{}, loc), k + 0) = reduce(
                LibTag{}, m_mesh, loc, (::dawn::float_type)0.000000,
                std::vector<dawn::LocationType>{
                    dawn::LocationType::Cells, dawn::LocationType::Edges, dawn::LocationType::Cells,
                    dawn::LocationType::Edges, dawn::LocationType::Cells},
                [&](auto& lhs, auto red_loc) {
                  lhs += m_in(deref(LibTag{}, red_loc), k + 0);
                  m_sparse_dimension_idx++;
                  return lhs;
                });
          }
        }
      }
      sync_storages();
    }
  };
  static constexpr const char* s_name = "intp";
  stencil_350 m_stencil_350;

public:
  reference_intp(const reference_intp&) = delete;

  // Members

  reference_intp(const dawn::mesh_t<LibTag>& mesh, int k_size,
                 dawn::cell_field_t<LibTag, double>& in, dawn::cell_field_t<LibTag, double>& out)
      : m_stencil_350(mesh, k_size, in, out) {}

  void run() {
    m_stencil_350.run();
    ;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
