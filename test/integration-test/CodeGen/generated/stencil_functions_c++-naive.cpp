// gtclang (0.0.1-b50903a-x86_64-linux-gnu-5.4.0)
// based on LLVM/Clang (3.8.0), Dawn (0.0.1)
// Generated on 2018-01-04  20:30:24

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

//
// Test 1
//

namespace cxxnaive {

class test_01_stencil {
 private:
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
            out(i + 0, j + 0, k + 0) = delta_i_plus_1_interval_start_0_end_0(
                i, j, k,
                ParamWrapper<gridtools::data_view<StorageType0>>(in, std::array<int, 3>{0, 0, 0} + in_offsets));
          }
        }
      }
    }
  };
  static constexpr const char* s_name = "test_01_stencil";
  sbase* m_stencil_0;

 public:
  test_01_stencil(const test_01_stencil&) = delete;

  // Members

  template <class StorageType1, class StorageType2>
  test_01_stencil(const gridtools::clang::domain& dom, StorageType1& in, StorageType2& out)
      : m_stencil_0(new stencil_0<StorageType1, StorageType2>(dom, in, out)) {}

  void run() {
    m_stencil_0->run();
    ;
  }
};
}  // namespace cxxnaiv
;

//
// Test 2
//

namespace cxxnaive {

class test_02_stencil {
 private:
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
            out(i + 0, j + 0, k + 0) =
                (delta_i_plus_1_interval_start_0_end_0(i, j, k, ParamWrapper<gridtools::data_view<StorageType0>>(
                                                                    in, std::array<int, 3>{0, 0, 0} + in_offsets)) +
                 delta_j_plus_1_interval_start_0_end_0(i, j, k, ParamWrapper<gridtools::data_view<StorageType0>>(
                                                                    in, std::array<int, 3>{0, 0, 0} + in_offsets)));
          }
        }
      }
    }
  };
  static constexpr const char* s_name = "test_02_stencil";
  sbase* m_stencil_0;

 public:
  test_02_stencil(const test_02_stencil&) = delete;

  // Members

  template <class StorageType1, class StorageType2>
  test_02_stencil(const gridtools::clang::domain& dom, StorageType1& in, StorageType2& out)
      : m_stencil_0(new stencil_0<StorageType1, StorageType2>(dom, in, out)) {}

  void run() {
    m_stencil_0->run();
    ;
  }
};
}  // namespace cxxnaiv
;

    ////
    //// Test 3
    ////

    ;

namespace cxxnaive {

class test_03_stencil {
 private:
  template <class StorageType0>
  static double delta_nested_i_plus_1_interval_start_0_end_0(const int i, const int j, const int k,
                                                             ParamWrapper<gridtools::data_view<StorageType0>> pw_in) {
    gridtools::data_view<StorageType0> in = pw_in.dview_;
    auto in_offsets = pw_in.offsets_;
    return delta_i_plus_1_interval_start_0_end_0(
        i, j, k, ParamWrapper<gridtools::data_view<StorageType0>>(in, std::array<int, 3>{0, 0, 0} + in_offsets));
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
            out(i + 0, j + 0, k + 0) = delta_nested_i_plus_1_interval_start_0_end_0(
                i, j, k,
                ParamWrapper<gridtools::data_view<StorageType0>>(in, std::array<int, 3>{0, 0, 0} + in_offsets));
          }
        }
      }
    }
  };
  static constexpr const char* s_name = "test_03_stencil";
  sbase* m_stencil_0;

 public:
  test_03_stencil(const test_03_stencil&) = delete;

  // Members

  template <class StorageType1, class StorageType2>
  test_03_stencil(const gridtools::clang::domain& dom, StorageType1& in, StorageType2& out)
      : m_stencil_0(new stencil_0<StorageType1, StorageType2>(dom, in, out)) {}

  void run() {
    m_stencil_0->run();
    ;
  }
};
}  // namespace cxxnaiv
;

    ////
    //// Test 6
    ////

    ;

;

;

namespace cxxnaive {

class test_06_stencil {
 private:
  template <class StorageType0>
  static double layer_3_ret_interval_start_0_end_0(const int i, const int j, const int k,
                                                   ParamWrapper<gridtools::data_view<StorageType0>> pw_in) {
    gridtools::data_view<StorageType0> in = pw_in.dview_;
    auto in_offsets = pw_in.offsets_;
    return layer_2_ret_interval_start_0_end_0(
        i, j, k, ParamWrapper<gridtools::data_view<StorageType0>>(in, std::array<int, 3>{0, 0, 0} + in_offsets));
  }

  template <class StorageType0>
  static double layer_2_ret_interval_start_0_end_0(const int i, const int j, const int k,
                                                   ParamWrapper<gridtools::data_view<StorageType0>> pw_in) {
    gridtools::data_view<StorageType0> in = pw_in.dview_;
    auto in_offsets = pw_in.offsets_;
    return layer_1_ret_interval_start_0_end_0(
        i, j, k, ParamWrapper<gridtools::data_view<StorageType0>>(in, std::array<int, 3>{0, 0, 0} + in_offsets));
  }

  template <class StorageType0>
  static double layer_1_ret_interval_start_0_end_0(const int i, const int j, const int k,
                                                   ParamWrapper<gridtools::data_view<StorageType0>> pw_in) {
    gridtools::data_view<StorageType0> in = pw_in.dview_;
    auto in_offsets = pw_in.offsets_;
    return in(i + 0 + in_offsets[0], j + 0 + in_offsets[1], k + 0 + in_offsets[2]);
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
            out(i + 0, j + 0, k + 0) = layer_3_ret_interval_start_0_end_0(
                i, j, k,
                ParamWrapper<gridtools::data_view<StorageType0>>(in, std::array<int, 3>{0, 0, 0} + in_offsets));
          }
        }
      }
    }
  };
  static constexpr const char* s_name = "test_06_stencil";
  sbase* m_stencil_0;

 public:
  test_06_stencil(const test_06_stencil&) = delete;

  // Members

  template <class StorageType1, class StorageType2>
  test_06_stencil(const gridtools::clang::domain& dom, StorageType1& in, StorageType2& out)
      : m_stencil_0(new stencil_0<StorageType1, StorageType2>(dom, in, out)) {}

  void run() {
    m_stencil_0->run();
    ;
  }
};
}  // namespace cxxnaiv
;

    ////
    //// Test 7
    ////

    ;

;

;

namespace cxxnaive {

class test_07_stencil {
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
      for (int k = 0 + 0; k <= (m_dom.ksize() == 0 ? 0 : (m_dom.ksize() - m_dom.kplus() - 1)) + 0; ++k) {
        for (int i = m_dom.iminus() + 0; i <= m_dom.isize() - m_dom.iplus() - 1 + 0; ++i) {
          for (int j = m_dom.jminus() + 0; j <= m_dom.jsize() - m_dom.jplus() - 1 + 0; ++j) {
            out(i + 0, j + 0, k + 0) = in(i + 0, j + 0, k + 0);
          }
        }
      }
    }
  };
  static constexpr const char* s_name = "test_07_stencil";
  sbase* m_stencil_0;

 public:
  test_07_stencil(const test_07_stencil&) = delete;

  // Members

  template <class StorageType1, class StorageType2>
  test_07_stencil(const gridtools::clang::domain& dom, StorageType1& in, StorageType2& out)
      : m_stencil_0(new stencil_0<StorageType1, StorageType2>(dom, in, out)) {}

  void run() {
    m_stencil_0->run();
    ;
  }
};
}  // namespace cxxnaiv
;
