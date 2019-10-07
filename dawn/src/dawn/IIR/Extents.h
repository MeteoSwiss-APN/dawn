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

#ifndef DAWN_IIR_EXTENTS_H
#define DAWN_IIR_EXTENTS_H

#include "dawn/IIR/LoopOrder.h"
#include "dawn/Support/Array.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/HashCombine.h"
#include <array>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iosfwd>
#include <optional>
#include <vector>

namespace dawn {
namespace iir {

/// @brief Access extent of a single dimension
/// @ingroup optimizer
struct Extent {
  int Minus;
  int Plus;

  /// @name Constructors and Assignment
  /// @{
  Extent() : Minus(0), Plus(0) {}
  Extent(int minus, int plus) : Minus(minus), Plus(plus) {}
  Extent(const Extent&) = default;
  Extent(Extent&&) = default;
  Extent& operator=(const Extent&) = default;
  Extent& operator=(Extent&&) = default;
  /// @}

  /// @name Operators
  /// @{
  Extent& merge(const Extent& other) {
    Minus = std::min(Minus, other.Minus);
    Plus = std::max(Plus, other.Plus);
    return *this;
  }

  Extent& expand(const Extent& other) {
    Minus = Minus + other.Minus;
    Plus = Plus + other.Plus;
    return *this;
  }

  Extent& merge(int other) {
    Minus = std::min(Minus, other < 0 ? other : 0);
    Plus = std::max(Plus, other > 0 ? other : 0);
    return *this;
  }

  Extent& add(const Extent& other) {
    Minus += other.Minus;
    Plus += other.Plus;
    return *this;
  }

  Extent& add(int other) {
    Minus += other;
    Plus += other;
    Minus = std::min(0, Minus);
    Plus = std::max(0, Plus);
    return *this;
  }

  static Extent add(const Extent& lhs, const Extent& rhs) {
    Extent sum;
    sum.Minus = lhs.Minus + rhs.Minus;
    sum.Plus = lhs.Plus + rhs.Plus;
    return sum;
  }

  bool operator==(const Extent& other) const { return Minus == other.Minus && Plus == other.Plus; }
  bool operator!=(const Extent& other) const { return !(*this == other); }
  /// @}
};

/// @brief Three dimensional access extents of a field
/// @ingroup optimizer
class Extents {
public:
  enum class VerticalLoopOrderDir { VL_CounterLoopOrder, VL_InLoopOrder };

  /// @name Constructors and Assignment
  /// @{
  explicit Extents(const Array3i& offset);
  Extents(int extent1minus, int extent1plus, int extent2minus, int extent2plus, int extent3minus,
          int extent3plus);
  Extents() = delete;
  Extents(const Extents&) = default;
  Extents(Extents&&) = default;
  Extents& operator=(const Extents&) = default;
  Extents& operator=(Extents&&) = default;
  /// @}

  /// @brief Get the i-th extend or NullExtent if i-th extent does not exists
  Extent& operator[](int i) { return extents_[i]; }
  const Extent& operator[](int i) const { return extents_[i]; }

  /// @brief Get the extents
  const std::array<Extent, 3>& getExtents() const { return extents_; }
  std::array<Extent, 3>& getExtents() { return extents_; }

  /// @brief Get size of extents (i.e number of dimensions)
  std::array<Extent, 3>::size_type getSize() const { return extents_.size(); }

  bool hasVerticalCenter() const { return extents_[2].Minus <= 0 && extents_[2].Plus >= 0; }

  /// @brief Merge `this` with `other` and assign an Extents to `this` which is the union of the two
  ///
  /// @b Example:
  ///   If `this` is `{-1, 1, 0, 0, 0, 0}` and `other` is `{-2, 0, 0, 0, 1}` the result will be
  ///   `{-2, 1, 0, 0, 0, 1}`.
  void merge(const Extents& other);
  void merge(const Array3i& offset);

  void expand(const Extents& other);

  /// @brief Add `this` and `other` and compute the direction sum of the two
  ///
  /// @b Example:
  ///   If `this` is `{-1, 1, -1, 1, 0, 0}` and `other` is `{0, 1, 0, 0, 0, 0}` the result will be
  ///   `{-1, 2, -1, 1, 0, 0}`.
  void add(const Array3i& offset);
  void add(const Extents& other);
  static Extents add(const Extents& lhs, const Extents& rhs);

  /// @brief Check if Extent is empty
  bool empty();

  /// @brief add the center of the stencil to the extent
  void addCenter(const unsigned int dim);

  /// @brief Check if extent in is pointwise (i.e equal to `{0, 0, 0, 0, 0, 0}`)
  bool isPointwise() const;

  /// @brief Check if extent in `dim` is pointwise (i.e equal to `{0, 0}`)
  bool isPointwiseInDim(int dim) const;

  /// @brief Check if this is a pointwise extent in the horizontal (i.e first two Extents are equal
  /// to {0, 0})
  bool isHorizontalPointwise() const;

  /// @brief Check if this is a pointwise extent in the vertical (i.e the third Extent is `{0, 0}`)
  bool isVerticalPointwise() const;

  struct VerticalLoopOrderAccess {
    bool CounterLoopOrder; ///< Access in the counter loop order
    bool LoopOrder;        ///< Access in the loop order
  };

  /// @brief Check if there is a stencil extent (i.e non-pointwise) in the counter-loop- and loop
  /// order
  ///
  /// If the loop order is forward, positive extents (e.g `k+1`) in the vertical are treated as
  /// counter-loop order accesses while negative extents (e.g `k-1`) are considered loop order
  /// accesses and vice versa for backward loop order. If the loop order is parallel, any
  /// non-pointwise extent is considered a counter-loop- and loop order access.
  VerticalLoopOrderAccess getVerticalLoopOrderAccesses(LoopOrderKind loopOrder) const;

  /// @brief Computes the fraction of this Extent being accessed in a LoopOrder or CounterLoopOrder
  /// return non initialized optional<Extent> if the full extent is accessed in a counterloop order
  /// manner
  /// @param loopOrder specifies the vertical loop order direction (forward, backward, or parallel)
  /// @param loopOrderPolicy specifies the requested policy for the requested extent access:
  ///            InLoopOrder/CounterLoopOrder
  /// @param includeCenter determines whether center is considered part of the loopOrderPolicy
  std::optional<Extent> getVerticalLoopOrderExtent(LoopOrderKind loopOrder,
                                                   VerticalLoopOrderDir loopOrderPolicy,
                                                   bool includeCenter) const;

  /// @brief format extents in string
  std::string toString() const;

  /// @brief Convert to stream
  friend std::ostream& operator<<(std::ostream& os, const Extents& extent);

  /// @brief Comparison operators
  /// @{
  bool operator==(const Extents& other) const;
  bool operator!=(const Extents& other) const;
  /// @}
private:
  std::array<Extent, 3> extents_;
};

} // namespace iir
} // namespace dawn

namespace std {

template <>
struct hash<dawn::iir::Extents> {
  size_t operator()(const dawn::iir::Extents& extent) const {
    size_t seed = 0;
    dawn::hash_combine(seed, extent[0].Minus, extent[0].Plus, extent[1].Minus, extent[1].Plus,
                       extent[2].Minus, extent[2].Plus);
    return seed;
  }
};

} // namespace std

#endif
