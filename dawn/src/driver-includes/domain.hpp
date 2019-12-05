//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _
//                         | |
//                       __| | __ ___      ___ ___
//                      / _` |/ _` \ \ /\ / / '_  |
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#pragma once

#include "driver-includes/halo.hpp"
#include <array>
#include <type_traits>

namespace gridtools {
namespace dawn {

/**
 * @brief 3-dimensional domain definition
 * @ingroup gridtools_dawn
 */
class domain {
  std::array<unsigned int, 3> m_dims;
  std::array<unsigned int, 6> m_halos;

public:
  /**
   * @brief Set size of domain
   *
   * To cancel a dimension, set its size to 1.
   *
   * @param i   Size of the first dimension
   * @param j   Size of the second dimension
   * @param k   Size of the third dimension
   */
  domain(unsigned int i, unsigned int j, unsigned int k)
      : m_dims{i, j, k}, m_halos{halo::value, halo::value, halo::value, halo::value, 0, 0} {}
  domain(const std::array<unsigned int, 3>& dims)
      : m_dims(dims), m_halos{halo::value, halo::value, halo::value, halo::value, 0, 0} {}

  domain(const domain&) = default;
  domain(domain&&) = default;
  domain& operator=(const domain&) = default;
  domain& operator=(domain&&) = default;

  /**
   * @brief Set the halo boundaries
   */
  void set_halos(unsigned int iminus = 0, unsigned int iplus = 0, unsigned int jminus = 0,
                 unsigned int jplus = 0, unsigned int kminus = 0, unsigned int kplus = 0) {
    m_halos = {iminus, iplus, jminus, jplus, kminus, kplus};
  }

  /**
   * @brief Get sizes
   */
  unsigned int isize() const { return m_dims[0]; }
  unsigned int jsize() const { return m_dims[1]; }
  unsigned int ksize() const { return m_dims[2]; }

  /**
   * @brief Get halo boundaries
   */
  unsigned int iminus() const { return m_halos[0]; }
  unsigned int iplus() const { return m_halos[1]; }
  unsigned int jminus() const { return m_halos[2]; }
  unsigned int jplus() const { return m_halos[3]; }
  unsigned int kminus() const { return m_halos[4]; }
  unsigned int kplus() const { return m_halos[5]; }

  /**
   * @brief Get dimension array
   */
  const std::array<unsigned int, 3>& dims() const { return m_dims; }

  /**
   * @brief Get halo boundary array
   */
  const std::array<unsigned int, 6>& halos() const { return m_halos; }
};

template <typename T>
struct is_domain : std::false_type {};

template <>
struct is_domain<domain> : std::true_type {};
} // namespace dawn
} // namespace gridtools
