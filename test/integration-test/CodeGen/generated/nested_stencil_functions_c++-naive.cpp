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

;

////
//// Test 4
////

;

;

namespace cxxnaive {

class test_04_stencil {
 private:
  template <class StorageType0>
  static double delta_sum_i_plus_1_j_plus_1_interval_start_0_end_0(
      const int i, const int j, const int k, ParamWrapper<gridtools::data_view<StorageType0>> pw_in0) {
    gridtools::data_view<StorageType0> in0 = pw_in0.dview_;
    auto in0_offsets = pw_in0.offsets_;
    return sum_delta_i_plus_1_interval_start_0_end_0_delta_j_plus_1_interval_start_0_end_0_interval_start_0_end_0(
        i, j, k, ParamWrapper<gridtools::data_view<StorageType0>>(in0, std::array<int, 3>{0, 0, 0} + in0_offsets),
        ParamWrapper<gridtools::data_view<StorageType0>>(in0, std::array<int, 3>{0, 0, 0} + in0_offsets));
  }

  template <class StorageType0, class StorageType1>
  static double sum_delta_i_plus_1_interval_start_0_end_0_delta_j_plus_1_interval_start_0_end_0_interval_start_0_end_0(
      const int i, const int j, const int k, ParamWrapper<gridtools::data_view<StorageType0>> pw_s1,
      ParamWrapper<gridtools::data_view<StorageType1>> pw_s2) {
    gridtools::data_view<StorageType0> s1 = pw_s1.dview_;
    auto s1_offsets = pw_s1.offsets_;
    gridtools::data_view<StorageType1> s2 = pw_s2.dview_;
    auto s2_offsets = pw_s2.offsets_;
    return (delta_i_plus_1_interval_start_0_end_0(i, j, k, pw_s1.cloneWithOffset(std::array<int, 3>{0, 0, 0})) +
            delta_j_plus_1_interval_start_0_end_0(i, j, k, pw_s1.cloneWithOffset(std::array<int, 3>{0, 0, 0})));
  }

  template <class StorageType0>
  static double delta_i_plus_1_interval_start_0_end_0(const int i, const int j, const int k,
                                                      ParamWrapper<gridtools::data_view<StorageType0>> pw_in) {
    gridtools::data_view<StorageType0> in = pw_in.dview_;
    auto in_offsets = pw_in.offsets_;
    return (in(i + 1 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2]) -
            in(i + 0 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2]));
  }

  template <class StorageType0>
  static double delta_j_plus_1_interval_start_0_end_0(const int i, const int j, const int k,
                                                      ParamWrapper<gridtools::data_view<StorageType0>> pw_in) {
    gridtools::data_view<StorageType0> in = pw_in.dview_;
    auto in_offsets = pw_in.offsets_;
    return (in(i + 0 + in_offsets[0], j + 1 + in_offsets[1], k + 0 + in_offsets[2]) -
            in(i + 0 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2]));
  }

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
      for (int k = 0 + 0; k <= (m_dom.ksize() == 0 ? 0 : (m_dom.ksize() - m_dom.kplus() - 1)) + 0; ++k) {
        for (int i = m_dom.iminus() + 0; i <= m_dom.isize() - m_dom.iplus() - 1 + 0; ++i) {
          for (int j = m_dom.jminus() + 0; j <= m_dom.jsize() - m_dom.jplus() - 1 + 0; ++j) {
            out(i + 0, j + 0, k + 0) = delta_sum_i_plus_1_j_plus_1_interval_start_0_end_0(
                i, j, k,
                ParamWrapper<gridtools::data_view<StorageType0>>(in, std::array<int, 3>{0, 0, 0} + in_offsets));
          }
        }
      }
    }
  };
  static constexpr const char* s_name = "test_04_stencil";
  sbase* m_stencil_0;

 public:
  test_04_stencil(const test_04_stencil&) = delete;

  // Members

  template <class StorageType1, class StorageType2>
  test_04_stencil(const gridtools::clang::domain& dom, StorageType1& in, StorageType2& out)
      : m_stencil_0(new stencil_0<StorageType1, StorageType2>(dom, in, out)) {}

  void run() {
    m_stencil_0->run();
    ;
  }
};
}  // namespace cxxnaiv
;

////
//// Test 5
////

namespace cxxnaive {

class test_05_stencil {
 private:
  template <class StorageType0>
  static double
  delta_i_plus_1_delta_j_plus_1_delta_i_plus_1_interval_start_0_end_0_interval_start_0_end_0_interval_start_0_end_0(
      const int i, const int j, const int k, ParamWrapper<gridtools::data_view<StorageType0>> pw_in) {
    gridtools::data_view<StorageType0> in = pw_in.dview_;
    auto in_offsets = pw_in.offsets_;
    return (delta_j_plus_1_delta_i_plus_1_interval_start_0_end_0_interval_start_0_end_0(
                i, j, k, pw_in.cloneWithOffset(std::array<int, 3>{0, 1, 0})) -
            delta_j_plus_1_delta_i_plus_1_interval_start_0_end_0_interval_start_0_end_0(
                i, j, k, pw_in.cloneWithOffset(std::array<int, 3>{0, 0, 0})));
  }

  template <class StorageType0>
  static double delta_j_plus_1_delta_i_plus_1_interval_start_0_end_0_interval_start_0_end_0(
      const int i, const int j, const int k, ParamWrapper<gridtools::data_view<StorageType0>> pw_in) {
    gridtools::data_view<StorageType0> in = pw_in.dview_;
    auto in_offsets = pw_in.offsets_;
    return (delta_i_plus_1_interval_start_0_end_0(i, j, k, pw_in.cloneWithOffset(std::array<int, 3>{1, 0, 0})) -
            delta_i_plus_1_interval_start_0_end_0(i, j, k, pw_in.cloneWithOffset(std::array<int, 3>{0, 0, 0})));
  }

  template <class StorageType0>
  static double delta_i_plus_1_interval_start_0_end_0(const int i, const int j, const int k,
                                                      ParamWrapper<gridtools::data_view<StorageType0>> pw_in) {
    gridtools::data_view<StorageType0> in = pw_in.dview_;
    auto in_offsets = pw_in.offsets_;
    return (in(i + 1 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2]) -
            in(i + 0 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2]));
  }

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
      for (int k = 0 + 0; k <= (m_dom.ksize() == 0 ? 0 : (m_dom.ksize() - m_dom.kplus() - 1)) + 0; ++k) {
        for (int i = m_dom.iminus() + 0; i <= m_dom.isize() - m_dom.iplus() - 1 + 0; ++i) {
          for (int j = m_dom.jminus() + 0; j <= m_dom.jsize() - m_dom.jplus() - 1 + 0; ++j) {
            out(i + 0, j + 0, k + 0) =
                delta_i_plus_1_delta_j_plus_1_delta_i_plus_1_interval_start_0_end_0_interval_start_0_end_0_interval_start_0_end_0(
                    i, j, k,
                    ParamWrapper<gridtools::data_view<StorageType0>>(in, std::array<int, 3>{0, 0, 0} + in_offsets));
          }
        }
      }
    }
  };
  static constexpr const char* s_name = "test_05_stencil";
  sbase* m_stencil_0;

 public:
  test_05_stencil(const test_05_stencil&) = delete;

  // Members

  template <class StorageType1, class StorageType2>
  test_05_stencil(const gridtools::clang::domain& dom, StorageType1& in, StorageType2& out)
      : m_stencil_0(new stencil_0<StorageType1, StorageType2>(dom, in, out)) {}

  void run() {
    m_stencil_0->run();
    ;
  }
};
}  // namespace cxxnaiv
;
