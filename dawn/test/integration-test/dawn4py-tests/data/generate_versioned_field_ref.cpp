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
class generate_versioned_field {
private:
  struct stencil_37 {
    ::dawn::mesh_t<LibTag> const& m_mesh;
    int m_k_size;
    ::dawn::edge_field_t<LibTag, ::dawn::float_type>& m_a;
    ::dawn::edge_field_t<LibTag, ::dawn::float_type>& m_b;
    ::dawn::edge_field_t<LibTag, ::dawn::float_type>& m_c;
    ::dawn::edge_field_t<LibTag, ::dawn::float_type>& m_d;
    ::dawn::edge_field_t<LibTag, ::dawn::float_type>& m_e;
    ::dawn::unstructured_domain m_unstructured_domain;

  public:
    stencil_37(::dawn::mesh_t<LibTag> const& mesh, int k_size,
               ::dawn::edge_field_t<LibTag, ::dawn::float_type>& a,
               ::dawn::edge_field_t<LibTag, ::dawn::float_type>& b,
               ::dawn::edge_field_t<LibTag, ::dawn::float_type>& c,
               ::dawn::edge_field_t<LibTag, ::dawn::float_type>& d,
               ::dawn::edge_field_t<LibTag, ::dawn::float_type>& e)
        : m_mesh(mesh), m_k_size(k_size), m_a(a), m_b(b), m_c(c), m_d(d), m_e(e) {}

    ~stencil_37() {}

    void sync_storages() {}
    static constexpr ::dawn::driver::unstructured_extent a_extent = {false, 0, 0};
    static constexpr ::dawn::driver::unstructured_extent b_extent = {false, 0, 0};
    static constexpr ::dawn::driver::unstructured_extent c_extent = {false, 0, 0};
    static constexpr ::dawn::driver::unstructured_extent d_extent = {false, 0, 0};
    static constexpr ::dawn::driver::unstructured_extent e_extent = {false, 0, 0};

    void run() {
      using ::dawn::deref;
      {
        for(int k = 0 + 0; k <= (m_k_size == 0 ? 0 : (m_k_size)) + 0 - 1 + 0; ++k) {
          for(auto const& loc : getEdges(LibTag{}, m_mesh)) {
            m_a(deref(LibTag{}, loc), (k + 0)) =
                ((m_b(deref(LibTag{}, loc), (k + 0)) / m_c(deref(LibTag{}, loc), (k + 0))) +
                 (::dawn::float_type)5);
          }
          for(auto const& loc : getEdges(LibTag{}, m_mesh)) {
            if(m_d(deref(LibTag{}, loc), (k + 0))) {
              m_a(deref(LibTag{}, loc), (k + 0)) = m_b(deref(LibTag{}, loc), (k + 0));
            } else {
              if(m_e(deref(LibTag{}, loc), (k + 0))) {
                m_c(deref(LibTag{}, loc), (k + 0)) =
                    (m_a(deref(LibTag{}, loc), (k + 0)) + (::dawn::float_type)1);
              }
            }
          }
        }
      }
      sync_storages();
    }
  };
  static constexpr const char* s_name = "generate_versioned_field";
  stencil_37 m_stencil_37;

public:
  generate_versioned_field(const generate_versioned_field&) = delete;

  // Members

  void set_splitter_index(::dawn::LocationType loc, ::dawn::UnstructuredSubdomain subdomain,
                          int offset, int index) {
    m_stencil_37.m_unstructured_domain.set_splitter_index({loc, subdomain, offset}, index);
  }

  generate_versioned_field(const ::dawn::mesh_t<LibTag>& mesh, int k_size,
                           ::dawn::edge_field_t<LibTag, ::dawn::float_type>& a,
                           ::dawn::edge_field_t<LibTag, ::dawn::float_type>& b,
                           ::dawn::edge_field_t<LibTag, ::dawn::float_type>& c,
                           ::dawn::edge_field_t<LibTag, ::dawn::float_type>& d,
                           ::dawn::edge_field_t<LibTag, ::dawn::float_type>& e)
      : m_stencil_37(mesh, k_size, a, b, c, d, e) {}

  void run() {
    m_stencil_37.run();
    ;
  }
};
} // namespace cxxnaiveico
} // namespace dawn_generated
