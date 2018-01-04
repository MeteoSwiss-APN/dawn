// gtclang (0.0.1-b50903a-x86_64-linux-gnu-5.4.0)
// based on LLVM/Clang (3.8.0), Dawn (0.0.1)
// Generated on 2018-01-04  20:30:25

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

struct globals : public gridtools::clang::globals_impl<globals> {
  using base_t = gridtools::clang::globals_impl<globals>;

  struct var_compiletime_adapter : public base_t::variable_adapter_impl<int> {
    var_compiletime_adapter() : base_t::variable_adapter_impl<int>(int(2)) {}

    template <class ValueType>
    var_compiletime_adapter& operator=(ValueType&& value) {
      throw std::runtime_error("invalid assignment to constant variable 'var_compiletime'");
      return *this;
    }
  } var_compiletime;

  struct var_runtime_adapter : public base_t::variable_adapter_impl<int> {
    var_runtime_adapter() : base_t::variable_adapter_impl<int>(int(1)) {}

    template <class ValueType>
    var_runtime_adapter& operator=(ValueType&& value) {
      get_value() = value;
      return *this;
    }
  } var_runtime;
};
template <>
globals* gridtools::clang::globals_impl<globals>::s_instance = nullptr;
}  // namespace cxxnaiv
;

//
// Test 1
//
namespace cxxnaive {

class test_01_stencil {
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
            out(i + 0, j + 0, k + 0) = (in(i + 0, j + 0, k + 0) + globals::get().var_runtime);
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
    if ((globals::get().var_runtime == (int)1)) {
      m_stencil_0->run();
    };
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
            out(i + 0, j + 0, k + 0) = (in(i + 0, j + 0, k + 0) + (int)2);
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

//
// Test 3
//
namespace cxxnaive {

class test_03_stencil {
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
            out(i + 0, j + 0, k + 0) = ((in(i + 0, j + 0, k + 0) + globals::get().var_runtime) + (int)2);
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
    if ((globals::get().var_runtime == (int)1)) {
      { m_stencil_0->run(); }
    };
  }
};
}  // namespace cxxnaiv
;

//
// Test 4
//
namespace cxxnaive {

class test_04_stencil {
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
            out(i + 0, j + 0, k + 0) = (gridtools::clang::float_type)0;
          }
        }
        for (int i = m_dom.iminus() + 0; i <= m_dom.isize() - m_dom.iplus() - 1 + 0; ++i) {
          for (int j = m_dom.jminus() + 0; j <= m_dom.jsize() - m_dom.jplus() - 1 + 0; ++j) {
            out(i + 0, j + 0, k + 0) += ((int)2 + in(i + 0, j + 0, k + 0));
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

//
// Test 5
//
namespace cxxnaive {

class test_05_stencil {
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
            out(i + 0, j + 0, k + 0) = ((int)2 * in(i + 0, j + 0, k + 0));
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
    gridtools::clang::float_type __local_some_var_4 = (gridtools::clang::float_type)5;
    ;
    if ((globals::get().var_runtime < __local_some_var_4)) {
      m_stencil_0->run();
    };
  }
};
}  // namespace cxxnaiv
;

//
// Test 6
//
namespace cxxnaive {

class test_06_stencil {
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
            out(i + 0, j + 0, k + 0) = ((int)2 * in(i + 0, j + 0, k + 0));
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
    gridtools::clang::float_type __local_some_var_4 = (gridtools::clang::float_type)5;
    ;
    m_stencil_0->run();
    ;
  }
};
}  // namespace cxxnaiv
;

//
// Test 7
//
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
            out(i + 0, j + 0, k + 0) = ((int)2 * in(i + 0, j + 0, k + 0));
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
    gridtools::clang::float_type __local_some_var_4 = (gridtools::clang::float_type)5;
    ;
    gridtools::clang::float_type __local_some_other_var_6 = (int)2;
    ;
    __local_some_var_4 += (gridtools::clang::float_type)1;
    ;
    m_stencil_0->run();
    ;
  }
};
}  // namespace cxxnaiv
;

//
// Test 8
//
namespace cxxnaive {

class test_08_stencil {
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
    gridtools::clang::meta_data_t m_meta_data;
    gridtools::clang::storage_t m_tmp;

   public:
    stencil_0(const gridtools::clang::domain& dom_, StorageType0& in_, StorageType1& out_)
        : m_dom(dom_),
          m_in(in_),
          m_out(out_),
          m_meta_data(dom_.isize(), dom_.jsize(), dom_.ksize()),
          m_tmp(m_meta_data) {}

    ~stencil_0() {}

    virtual void run() {
      gridtools::data_view<StorageType0> in = gridtools::make_host_view(m_in);
      std::array<int, 3> in_offsets{0, 0, 0};
      gridtools::data_view<StorageType1> out = gridtools::make_host_view(m_out);
      std::array<int, 3> out_offsets{0, 0, 0};
      gridtools::data_view<storage_t> tmp = gridtools::make_host_view(m_tmp);
      std::array<int, 3> tmp_offsets{0, 0, 0};
      for (int k = 0 + 0; k <= (m_dom.ksize() == 0 ? 0 : (m_dom.ksize() - m_dom.kplus() - 1)) + 0; ++k) {
        for (int i = m_dom.iminus() + 0; i <= m_dom.isize() - m_dom.iplus() - 1 + 0; ++i) {
          for (int j = m_dom.jminus() + 0; j <= m_dom.jsize() - m_dom.jplus() - 1 + 0; ++j) {
            tmp(i + 0, j + 0, k + 0) = ((int)2 * in(i + 0, j + 0, k + 0));
          }
        }
        for (int i = m_dom.iminus() + 0; i <= m_dom.isize() - m_dom.iplus() - 1 + 0; ++i) {
          for (int j = m_dom.jminus() + 0; j <= m_dom.jsize() - m_dom.jplus() - 1 + 0; ++j) {
            out(i + 0, j + 0, k + 0) = ((int)2 * tmp(i + 0, j + 0, k + 0));
          }
        }
      }
    }
  };
  static constexpr const char* s_name = "test_08_stencil";
  sbase* m_stencil_0;

 public:
  test_08_stencil(const test_08_stencil&) = delete;

  // Members

  template <class StorageType1, class StorageType2>
  test_08_stencil(const gridtools::clang::domain& dom, StorageType1& in, StorageType2& out)
      : m_stencil_0(new stencil_0<StorageType1, StorageType2>(dom, in, out)) {}

  void run() {
    m_stencil_0->run();
    ;
  }
};
}  // namespace cxxnaiv
;

//
// Test 9
//
namespace cxxnaive {

class test_09_stencil_call {
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
            out(i + 0, j + 0, k + 0) = ((int)2 * in(i + 0, j + 0, k + 0));
          }
        }
      }
    }
  };
  static constexpr const char* s_name = "test_09_stencil_call";
  sbase* m_stencil_0;

 public:
  test_09_stencil_call(const test_09_stencil_call&) = delete;

  // Members

  template <class StorageType1, class StorageType2>
  test_09_stencil_call(const gridtools::clang::domain& dom, StorageType1& in, StorageType2& out)
      : m_stencil_0(new stencil_0<StorageType1, StorageType2>(dom, in, out)) {}

  void run() {
    m_stencil_0->run();
    ;
  }
};
}  // namespace cxxnaiv
;

namespace cxxnaive {

class test_09_stencil {
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
            out(i + 0, j + 0, k + 0) = ((int)2 * in(i + 0, j + 0, k + 0));
          }
        }
      }
    }
  };
  static constexpr const char* s_name = "test_09_stencil";
  sbase* m_stencil_0;

 public:
  test_09_stencil(const test_09_stencil&) = delete;

  // Members

  template <class StorageType1, class StorageType2>
  test_09_stencil(const gridtools::clang::domain& dom, StorageType1& in, StorageType2& out)
      : m_stencil_0(new stencil_0<StorageType1, StorageType2>(dom, in, out)) {}

  void run() {
    m_stencil_0->run();
    ;
  }
};
}  // namespace cxxnaiv
;
