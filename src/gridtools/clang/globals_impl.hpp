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

#pragma once

#ifdef GRIDTOOLS_CLANG_GENERATED
#include "gridtools/clang/storage.hpp"
#include <memory>
#include <stdexcept>
#endif

namespace gridtools {

namespace clang {

#ifdef GRIDTOOLS_CLANG_GENERATED
/**
 * @brief Implementation of the global variables
 */
template <class Derived>
class globals_impl {
  static Derived* s_instance;

public:
  /**
   * @brief Wrapper of a variable of type `T`
   */
  template <class T>
  struct variable_storage {
    T value;
  };

  /**
   * @brief Wrapper class around `gridtools::global_parameter`
   *
   * The `global_parameter` is constructed lazily i.e upon first usage.
   */
  template <class T>
  class variable_adapter_impl {
  public:
    using global_parameter_t = storage_traits_t::data_store_t<
        variable_storage<T>, storage_traits_t::special_storage_info_t<0, gridtools::selector<0u>>>;

    /**
     * @brief Initialize the variable with the given `value`
     */
    template <class ValueType>
    variable_adapter_impl(ValueType&& value) : m_global_parameter_ptr(nullptr) {
      m_data.value = value;
    }

    /**
     * @brief Get an up-to-date refrence to the variable
     */
    T& get_value() {
      if(m_global_parameter_ptr)
        m_global_parameter_ptr->sync();
      return m_data.value;
    }

    /**
     * @brief Convert to type `T` (return a copy of the value)
     */
    operator T() { return get_value(); }

    /**
     * @brief Get a refrence to the `global_parameter`
     */
    global_parameter_t& as_global_parameter() {
      if(!m_global_parameter_ptr) {
        m_global_parameter_ptr = std::unique_ptr<global_parameter_t>(
            new global_parameter_t(backend_t::make_global_parameter(m_data)));
      }
      return *m_global_parameter_ptr;
    }

  private:
    variable_storage<T> m_data;
    std::unique_ptr<global_parameter_t> m_global_parameter_ptr;
  };

  ~globals_impl() { reset(); }

  static Derived& get() {
    if(!s_instance)
      s_instance = new Derived;
    return *s_instance;
  }

  static void reset() {
    if(s_instance)
      delete s_instance;
  }
};
#else
/**
 * @brief globals implementation
 */
template <class Derived>
class globals_impl {
public:
  static Derived& get();
  static void reset();
};
#endif
}
}
