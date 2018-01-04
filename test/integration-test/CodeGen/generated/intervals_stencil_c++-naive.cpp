// gtclang (0.0.1-b50903a-x86_64-linux-gnu-5.4.0)
// based on LLVM/Clang (3.8.0), Dawn (0.0.1)
// Generated on 2018-01-04  20:30:22

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

#ifndef GRIDTOOLS_CLANG_GENERATED
interval k_flat = k_start + 4;
#endif

// Check if we correclty generate the empty Do-Methods according to
// https://github.com/eth-cscs/gridtools/issues/330

namespace cxxnaive {

class intervals_stencil {
 private:
  struct sbase {
    virtual void run() {}

    virtual ~sbase() {}
  };
  template <class StorageType0, class StorageType1>
  struct stencil_0 : public sbase {
    // //Members
    const gridtools::clang::domain& m_dom;
    StorageType0& m_in;
    StorageType1& m_out;

   public:
    stencil_0(const gridtools::clang::domain& dom_, StorageType0& in_, StorageType1& out_)
        : m_dom(dom_), m_in(in_), m_out(out_) {}

    ~stencil_0() {}

    virtual void run() {
      gridtools::data_view<StorageType0> in = gridtools::make_host_view(m_in);
      std::array<int, 3> in_offsets{0, 0, 0};
      gridtools::data_view<StorageType1> out = gridtools::make_host_view(m_out);
      std::array<int, 3> out_offsets{0, 0, 0};
      for (int k = 0 + 0; k <= 4 + 0; ++k) {
        for (int i = m_dom.iminus() + 0; i <= m_dom.isize() - m_dom.iplus() - 1 + 0; ++i) {
          for (int j = m_dom.jminus() + 0; j <= m_dom.jsize() - m_dom.jplus() - 1 + 0; ++j) {
            out(i + 0, j + 0, k + 0) = (in(i + 0, j + 0, k + 0) + (int)1);
          }
        }
      }
      for (int k = 5 + 0; k <= 5 + 0; ++k) {
        for (int i = m_dom.iminus() + 0; i <= m_dom.isize() - m_dom.iplus() - 1 + 0; ++i) {
          for (int j = m_dom.jminus() + 0; j <= m_dom.jsize() - m_dom.jplus() - 1 + 0; ++j) {
            out(i + 0, j + 0, k + 0) = (in(i + 0, j + 0, k + 0) + (int)2);
          }
        }
      }
      for (int k = 6 + 0; k <= (m_dom.ksize() == 0 ? 0 : (m_dom.ksize() - m_dom.kplus() - 1)) + 0; ++k) {
        for (int i = m_dom.iminus() + 0; i <= m_dom.isize() - m_dom.iplus() - 1 + 0; ++i) {
          for (int j = m_dom.jminus() + 0; j <= m_dom.jsize() - m_dom.jplus() - 1 + 0; ++j) {
            out(i + 0, j + 0, k + 0) = (in(i + 0, j + 0, k + 0) + (int)3);
          }
        }
      }
    }
  };
  static constexpr const char* s_name = "intervals_stencil";
  sbase* m_stencil_0;

 public:
  intervals_stencil(const intervals_stencil&) = delete;

  // Members

  template <class StorageType1, class StorageType2>
  intervals_stencil(const gridtools::clang::domain& dom, StorageType1& in, StorageType2& out)
      : m_stencil_0(new stencil_0<StorageType1, StorageType2>(dom, in, out)) {}

  void run() {
    m_stencil_0->run();
    ;
  }
};
}  // namespace cxxnaiv
;
