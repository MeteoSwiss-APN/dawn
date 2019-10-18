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

#include "LoopOrder.h"

#include "dawn/AST/Offsets.h"
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
class Extent {
private:
  int Minus;
  int Plus;

public:
  /// @name Constructors and Assignment
  /// @{
  Extent() : Minus(0), Plus(0) {}
  Extent(int minus, int plus) : Minus(minus), Plus(plus) {}
  /// @}

  int minus() const { return Minus; }
  int plus() const { return Plus; }

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

  Extent& operator+=(const Extent& other) {
    Minus += other.Minus;
    Plus += other.Plus;
    return *this;
  }

  Extent& operator+=(int other) {
    Minus += other;
    Plus += other;
    Minus = std::min(0, Minus);
    Plus = std::max(0, Plus);
    return *this;
  }

  bool operator==(const Extent& other) const { return Minus == other.Minus && Plus == other.Plus; }
  bool operator!=(const Extent& other) const { return !(*this == other); }

  bool isPointwise() const { return Plus == 0 && Minus == 0; }
  /// @}
};
Extent operator+(Extent lhs, Extent const& rhs);

class HorizontalExtentImpl {
public:
  HorizontalExtentImpl() = default;
  virtual ~HorizontalExtentImpl() = default;

  HorizontalExtentImpl& operator+=(HorizontalExtentImpl const& other) {
    add_impl(other);
    return *this;
  }
  std::unique_ptr<HorizontalExtentImpl> clone() const { return clone_impl(); }

  void merge(HorizontalExtentImpl const& other) { merge_impl(other); }
  void merge(dawn::ast::HorizontalOffset const& other) { merge_impl(other); }
  void expand(HorizontalExtentImpl const& other) { expand_impl(other); }
  bool equals(HorizontalExtentImpl const& other) const { return equals_impl(other); }
  bool isPointwise() const { return isPointwise_impl(); }

protected:
  virtual void add_impl(HorizontalExtentImpl const& other) = 0;
  virtual void merge_impl(HorizontalExtentImpl const& other) = 0;
  virtual void merge_impl(dawn::ast::HorizontalOffset const& other) = 0;
  virtual void expand_impl(HorizontalExtentImpl const& other) = 0;
  virtual bool equals_impl(HorizontalExtentImpl const& other) const = 0;
  virtual std::unique_ptr<HorizontalExtentImpl> clone_impl() const = 0;
  virtual bool isPointwise_impl() const = 0;
};

class CartesianExtent : public HorizontalExtentImpl {
public:
  CartesianExtent(int iMinus, int iPlus, int jMinus, int jPlus) {
    m_extents_[0] = Extent(iMinus, iPlus);
    m_extents_[1] = Extent(jMinus, jPlus);
  }

  CartesianExtent() {
    m_extents_[0] = Extent(0, 0);
    m_extents_[1] = Extent(0, 0);
  }

  void add_impl(HorizontalExtentImpl const& other) override {
    auto other_cartesian = dynamic_cast<dawn::iir::CartesianExtent const&>(other);
    m_extents_[0] += other_cartesian.m_extents_[0];
    m_extents_[1] += other_cartesian.m_extents_[1];
  }

  void merge_impl(HorizontalExtentImpl const& other) override {
    auto other_cartesian = dynamic_cast<dawn::iir::CartesianExtent const&>(other);
    m_extents_[0].merge(other_cartesian.m_extents_[0]);
    m_extents_[1].merge(other_cartesian.m_extents_[1]);
  }

  void merge_impl(dawn::ast::HorizontalOffset const& offset) override {
    auto offset_cartesian = dawn::ast::offset_cast<dawn::ast::CartesianOffset const&>(offset);
    m_extents_[0].merge(Extent(offset_cartesian.offsetI(), offset_cartesian.offsetI()));
    m_extents_[1].merge(Extent(offset_cartesian.offsetJ(), offset_cartesian.offsetJ()));
  }

  void expand_impl(HorizontalExtentImpl const& other) override {
    auto other_cartesian = dynamic_cast<dawn::iir::CartesianExtent const&>(other);
    m_extents_[0].expand(other_cartesian.m_extents_[0]);
    m_extents_[1].expand(other_cartesian.m_extents_[1]);
  }

  bool equals_impl(HorizontalExtentImpl const& other) const override {
    auto other_cartesian = dynamic_cast<dawn::iir::CartesianExtent const&>(other);
    bool equalI = m_extents_[0] == other_cartesian.m_extents_[0];
    bool equalJ = m_extents_[1] == other_cartesian.m_extents_[1];
    return equalI && equalJ;
  }

  std::unique_ptr<HorizontalExtentImpl> clone_impl() const override {
    return std::unique_ptr<HorizontalExtentImpl>(new CartesianExtent(
        m_extents_[0].minus(), m_extents_[0].plus(), m_extents_[1].minus(), m_extents_[1].plus()));
  }

  bool isPointwise_impl() const override {
    return m_extents_[0].isPointwise() && m_extents_[1].isPointwise();
  }

  int iMinus() const { return m_extents_[0].minus(); }
  int iPlus() const { return m_extents_[0].plus(); }
  int jMinus() const { return m_extents_[1].minus(); }
  int jPlus() const { return m_extents_[1].plus(); }

private:
  std::array<Extent, 2> m_extents_;
};

class HorizontalExtent {
public:
  HorizontalExtent(ast::cartesian_) : impl_(std::make_unique<CartesianExtent>()) {}
  HorizontalExtent(ast::cartesian_, int iMinus, int iPlus, int jMinus, int jPlus)
      : impl_(std::make_unique<CartesianExtent>(iMinus, iPlus, jMinus, jPlus)) {}

  HorizontalExtent() = delete;
  HorizontalExtent(HorizontalExtent const& other) : impl_(other.impl_->clone()) {}
  HorizontalExtent(HorizontalExtent&& other) : impl_(std::move(other.impl_)) {}

  HorizontalExtent& operator=(HorizontalExtent const& other) {
    impl_ = other.impl_->clone();
    return *this;
  }
  HorizontalExtent& operator=(HorizontalExtent&& other) = default;

  HorizontalExtent(HorizontalExtentImpl* impl)
      : impl_(std::unique_ptr<HorizontalExtentImpl>(impl)) {}

  HorizontalExtent& operator+=(HorizontalExtent const& other) {
    *impl_ += *other.impl_;
    return *this;
  }

  template <typename T>
  friend T extent_cast(HorizontalExtent const&);
  template <typename T>
  friend T extent_cast(HorizontalExtent&&);

  bool operator==(HorizontalExtent const& other) const { return impl_->equals(*other.impl_); }
  bool operator!=(HorizontalExtent const& other) const { return !(*this == other); }
  void merge(const HorizontalExtent& other) { impl_->merge(*other.impl_); }
  void merge(const dawn::ast::HorizontalOffset& other) { impl_->merge(other); }
  void expand(const HorizontalExtent& other) { impl_->expand(*other.impl_); }
  bool isPointwise() const { return impl_->isPointwise(); }

private:
  std::unique_ptr<HorizontalExtentImpl> impl_;
};

template <typename T>
T extent_cast(HorizontalExtent const& extent) {
  return dynamic_cast<T>(*extent.impl_);
}
template <typename T>
T extent_cast(HorizontalExtent&& extent) {
  return dynamic_cast<T>(*extent.impl_);
}

/// @brief Three dimensional access extents of a field
/// @ingroup optimizer
class Extents {
public:
  enum class VerticalLoopOrderDir { VL_CounterLoopOrder, VL_InLoopOrder };

  /// @name Constructors and Assignment
  /// @{
  Extents(ast::cartesian_, const ast::Offsets& offset);
  Extents(ast::cartesian_, int extent1minus, int extent1plus, int extent2minus, int extent2plus,
          int extent3minus, int extent3plus);
  explicit Extents(ast::cartesian_);
  /// @}

  bool hasVerticalCenter() const;

  /// @brief Merge `this` with `other` and assign an Extents to `this` which is the union of the two
  ///
  /// @b Example:
  ///   If `this` is `{-1, 1, 0, 0, 0, 0}` and `other` is `{-2, 0, 0, 0, 1}` the result will be
  ///   `{-2, 1, 0, 0, 0, 1}`.
  void merge(const Extents& other);
  void merge(const ast::Offsets& offset);

  void expand(const Extents& other);

  /// @brief resets vertical extants to {0, 0}
  void resetVerticalExtent();

  /// @brief Check if extent is pointwise in the horizontal direction, i.e. zero extent
  bool isHorizontalPointwise() const;

  /// @brief Check if extent is pointwise in the horizontal direction, i.e. zero extent
  bool isVerticalPointwise() const;

  /// @brief Check if extent is pointwise all directions, i.e. zero extent
  bool isPointwise() const;

  /// @brief add the center of the stencil to the vertical extent
  ///   i.e. make sure that (0,0) is included in the vertical extent
  void addVerticalCenter();

  struct VerticalLoopOrderAccess {
    bool CounterLoopOrder; ///< Access in the counter loop order
    bool LoopOrder;        ///< Access in the loop order
  };

  VerticalLoopOrderAccess getVerticalLoopOrderAccess() const;

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
  /// @brief Convert to stream
  friend std::ostream& operator<<(std::ostream& os, const Extents& extent);
  std::string toString() const;

  /// @brief Comparison operators
  /// @{
  bool operator==(const Extents& other) const;
  bool operator!=(const Extents& other) const;
  /// @}

  Extent const& verticalExtent() const { return verticalExtent_; }
  HorizontalExtent const& horizontalExtent() const { return horizontalExtent_; }

  Extents& operator+=(const Extents& other);

private:
  // void add(const Extents& other);
  Extent verticalExtent_;
  HorizontalExtent horizontalExtent_;
};

/// @brief Add `this` and `other` and compute the direction sum of the two
///
/// @b Example:
///   If `this` is `{-1, 1, -1, 1, 0, 0}` and `other` is `{0, 1, 0, 0, 0, 0}` the result will be
///   `{-1, 2, -1, 1, 0, 0}`.
Extents operator+(Extents lhs, const Extents& rhs);

} // namespace iir
} // namespace dawn

namespace std {

template <>
struct hash<dawn::iir::Extents> {
  size_t operator()(const dawn::iir::Extents& extent) const {
    auto const& hextent =
        dawn::iir::extent_cast<dawn::iir::CartesianExtent const&>(extent.horizontalExtent());
    auto const& vextent = extent.verticalExtent();

    size_t seed = 0;
    dawn::hash_combine(seed, hextent.iMinus(), hextent.iPlus(), hextent.jMinus(), hextent.jPlus(),
                       vextent.minus(), vextent.plus());
    return seed;
  }
};

} // namespace std

#endif
