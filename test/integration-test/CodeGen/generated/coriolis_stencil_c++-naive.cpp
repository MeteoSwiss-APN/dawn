// gtclang (0.0.1-b9691ca-x86_64-linux-gnu-5.4.0)
// based on LLVM/Clang (3.8.0), Dawn (0.0.1)
// Generated on 2018-01-02  01:22:54

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

namespace cxxnaive {
template <size_t N>
std::array<int, N> operator+(std::array<int, N> const& a, std::array<int, N> const& b) {
  std::array<int, N> res;
  for (size_t i = 0; i < N; ++i) {
    res[i] = a[i] + b[i];
  }
  return res;
}

class coriolis_stencil {
 private:
  template <class DataView>
  struct ParamWrapper {
    DataView dview_;
    std::array<int, DataView::storage_info_t::ndims> offsets_;

    ParamWrapper(DataView dview, std::array<int, DataView::storage_info_t::ndims> offsets)
        : dview_(dview), offsets_(offsets) {}
  };

  struct sbase {
    virtual void run() {}

    virtual ~sbase() {}
  };
  template <class StorageType0, class StorageType1, class StorageType2, class StorageType3, class StorageType4>
  struct stencil_0 : public sbase {
    // //Members
    const gridtools::clang::domain& m_dom;
    StorageType0& m_u_tens;
    StorageType1& m_u_nnow;
    StorageType2& m_v_tens;
    StorageType3& m_v_nnow;
    StorageType4& m_fc;

   public:
    stencil_0(const gridtools::clang::domain& dom_, StorageType0& u_tens_, StorageType1& u_nnow_, StorageType2& v_tens_,
              StorageType3& v_nnow_, StorageType4& fc_)
        : m_dom(dom_), m_u_tens(u_tens_), m_u_nnow(u_nnow_), m_v_tens(v_tens_), m_v_nnow(v_nnow_), m_fc(fc_) {}

    ~stencil_0() {}

    virtual void run() {
      gridtools::data_view<StorageType0> u_tens = gridtools::make_host_view(m_u_tens);
      std::array<int, 3> u_tens_offsets{0, 0, 0};
      gridtools::data_view<StorageType1> u_nnow = gridtools::make_host_view(m_u_nnow);
      std::array<int, 3> u_nnow_offsets{0, 0, 0};
      gridtools::data_view<StorageType2> v_tens = gridtools::make_host_view(m_v_tens);
      std::array<int, 3> v_tens_offsets{0, 0, 0};
      gridtools::data_view<StorageType3> v_nnow = gridtools::make_host_view(m_v_nnow);
      std::array<int, 3> v_nnow_offsets{0, 0, 0};
      gridtools::data_view<StorageType4> fc = gridtools::make_host_view(m_fc);
      std::array<int, 3> fc_offsets{0, 0, 0};
      for (int k = 0 + 0; k <= (m_dom.ksize() == 0 ? 0 : (m_dom.ksize() - m_dom.kplus() - 1)) + 0; ++k) {
        for (int i = m_dom.iminus() + 0; i <= m_dom.isize() - m_dom.iplus() - 1 + 0; ++i) {
          for (int j = m_dom.jminus() + 0; j <= m_dom.jsize() - m_dom.jplus() - 1 + 0; ++j) {
            gridtools::clang::float_type __local_z_fv_north_8 =
                (fc(i + 0, j + 0, k + 0) * (v_nnow(i + 0, j + 0, k + 0) + v_nnow(i + 1, j + 0, k + 0)));
            gridtools::clang::float_type __local_z_fv_south_9 =
                (fc(i + 0, j + -1, k + 0) * (v_nnow(i + 0, j + -1, k + 0) + v_nnow(i + 1, j + -1, k + 0)));
            u_tens(i + 0, j + 0, k + 0) +=
                ((gridtools::clang::float_type)0.25 * (__local_z_fv_north_8 + __local_z_fv_south_9));
            gridtools::clang::float_type __local_z_fu_east_11 =
                (fc(i + 0, j + 0, k + 0) * (u_nnow(i + 0, j + 0, k + 0) + u_nnow(i + 0, j + 1, k + 0)));
            gridtools::clang::float_type __local_z_fu_west_12 =
                (fc(i + -1, j + 0, k + 0) * (u_nnow(i + -1, j + 0, k + 0) + u_nnow(i + -1, j + 1, k + 0)));
            v_tens(i + 0, j + 0, k + 0) -=
                ((gridtools::clang::float_type)0.25 * (__local_z_fu_east_11 + __local_z_fu_west_12));
          }
        }
      }
    }
  };
  static constexpr const char* s_name = "coriolis_stencil";
  sbase* m_stencil_0;

 public:
  coriolis_stencil(const coriolis_stencil&) = delete;

  // Members

  template <class StorageType1, class StorageType2, class StorageType3, class StorageType4, class StorageType5>
  coriolis_stencil(const gridtools::clang::domain& dom, StorageType1& u_tens, StorageType2& u_nnow,
                   StorageType3& v_tens, StorageType4& v_nnow, StorageType5& fc)
      : m_stencil_0(new stencil_0<StorageType1, StorageType2, StorageType3, StorageType4, StorageType5>(
            dom, u_tens, u_nnow, v_tens, v_nnow, fc)) {}

  void run() { m_stencil_0->run(); }
};
}  // namespace cxxnaiv
;
