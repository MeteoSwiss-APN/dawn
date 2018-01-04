// gtclang (0.0.1-b50903a-x86_64-linux-gnu-5.4.0)
// based on LLVM/Clang (3.8.0), Dawn (0.0.1)
// Generated on 2018-01-04  20:30:26

#define GRIDTOOLS_CLANG_GENERATED 1
#ifndef BOOST_RESULT_OF_USE_TR1
 #define BOOST_RESULT_OF_USE_TR1 1
#endif
#ifndef BOOST_NO_CXX11_DECLTYPE
 #define BOOST_NO_CXX11_DECLTYPE 1
#endif
//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _
//                        | |     | |
//                    __ _| |_ ___| | __ _ _ __   __ _
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

;

namespace cxxnaive {

class hori_diff_stencil {
 private:
  template <class StorageType0>
  static double laplacian_interval_start_0_end_0(const int i, const int j, const int k,
                                                 ParamWrapper<gridtools::data_view<StorageType0>> pw_phi) {
    gridtools::data_view<StorageType0> phi = pw_phi.dview_;
    auto phi_offsets = pw_phi.offsets_;
    return ((((phi(i + 1 + phi_offsets[0], j + 0 + phi_offsets[1], k + 0 + phi_offsets[2]) +
               phi(i + -1 + phi_offsets[0], j + 0 + phi_offsets[1], k + 0 + phi_offsets[2])) +
              phi(i + 0 + phi_offsets[0], j + 1 + phi_offsets[1], k + 0 + phi_offsets[2])) +
             phi(i + 0 + phi_offsets[0], j + -1 + phi_offsets[1], k + 0 + phi_offsets[2])) -
            ((gridtools::clang::float_type)4 *
             phi(i + 0 + phi_offsets[0], j + 0 + phi_offsets[1], k + 0 + phi_offsets[2])));
  }

  struct sbase {
    virtual void run() {}

    virtual ~sbase() {}
  };
  template <class StorageType0, class StorageType1, class StorageType2>
  struct stencil_0 : public sbase {
    // //Members
    const gridtools::clang::domain& m_dom;
    StorageType0& m_u;
    StorageType1& m_out;
    StorageType2& m_lap;

   public:
    stencil_0(const gridtools::clang::domain& dom_, StorageType0& u_, StorageType1& out_, StorageType2& lap_)
        : m_dom(dom_), m_u(u_), m_out(out_), m_lap(lap_) {}

    ~stencil_0() {}

    virtual void run() {
      gridtools::data_view<StorageType0> u = gridtools::make_host_view(m_u);
      std::array<int, 3> u_offsets{0, 0, 0};
      gridtools::data_view<StorageType1> out = gridtools::make_host_view(m_out);
      std::array<int, 3> out_offsets{0, 0, 0};
      gridtools::data_view<StorageType2> lap = gridtools::make_host_view(m_lap);
      std::array<int, 3> lap_offsets{0, 0, 0};
      for (int k = 0 + 0; k <= (m_dom.ksize() == 0 ? 0 : (m_dom.ksize() - m_dom.kplus() - 1)) + 0; ++k) {
        for (int i = m_dom.iminus() + -1; i <= m_dom.isize() - m_dom.iplus() - 1 + 1; ++i) {
          for (int j = m_dom.jminus() + -1; j <= m_dom.jsize() - m_dom.jplus() - 1 + 1; ++j) {
            lap(i + 0, j + 0, k + 0) = laplacian_interval_start_0_end_0(
                i, j, k, ParamWrapper<gridtools::data_view<StorageType0>>(u, std::array<int, 3>{0, 0, 0} + u_offsets));
          }
        }
        for (int i = m_dom.iminus() + 0; i <= m_dom.isize() - m_dom.iplus() - 1 + 0; ++i) {
          for (int j = m_dom.jminus() + 0; j <= m_dom.jsize() - m_dom.jplus() - 1 + 0; ++j) {
            out(i + 0, j + 0, k + 0) = laplacian_interval_start_0_end_0(
                i, j, k,
                ParamWrapper<gridtools::data_view<StorageType2>>(lap, std::array<int, 3>{0, 0, 0} + lap_offsets));
          }
        }
      }
    }
  };
  static constexpr const char* s_name = "hori_diff_stencil";
  sbase* m_stencil_0;

 public:
  hori_diff_stencil(const hori_diff_stencil&) = delete;

  // Members

  template <class StorageType1, class StorageType2, class StorageType3>
  hori_diff_stencil(const gridtools::clang::domain& dom, StorageType1& u, StorageType2& out, StorageType3& lap)
      : m_stencil_0(new stencil_0<StorageType1, StorageType2, StorageType3>(dom, u, out, lap)) {}

  void run() {
    m_stencil_0->run();
    ;
  }
};
}  // namespace cxxnaiv
;
