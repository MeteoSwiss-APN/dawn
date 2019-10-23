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

class Extents;

/// @brief Access extent of a single dimension
/// @ingroup optimizer
class Extent {

public:
  /// @name Constructors and Assignment
  /// @{
  Extent(int minus, int plus) : minus_(minus), plus_(plus) {}
  Extent() : Extent(0, 0) {}
  /// @}

  int minus() const { return minus_; }
  int plus() const { return plus_; }

  /// @name Operators
  /// @{
  Extent& merge(const Extent& other) {
    minus_ = std::min(minus_, other.minus_);
    plus_ = std::max(plus_, other.plus_);
    return *this;
  }
  Extent& merge(int other) { return merge(Extent{other, other}); }

  Extent limit(int minus, int plus) const {
    DAWN_ASSERT(minus <= 0 && plus >= 0);
    return {std::max(minus, minus_), std::min(plus, plus_)};
  }

  Extent& operator+=(const Extent& other) {
    minus_ += other.minus_;
    plus_ += other.plus_;
    return *this;
  }

  bool operator==(const Extent& other) const {
    return minus_ == other.minus_ && plus_ == other.plus_;
  }
  bool operator!=(const Extent& other) const { return !(*this == other); }

  bool isPointwise() const { return plus_ == 0 && minus_ == 0; }
  /// @}
  //
private:
  int minus_;
  int plus_;
};
Extent operator+(Extent lhs, Extent const& rhs);

class HorizontalExtentImpl {
public:
  HorizontalExtentImpl() = default;
  virtual ~HorizontalExtentImpl() = default;

  HorizontalExtentImpl& operator+=(HorizontalExtentImpl const& other) {
    addImpl(other);
    return *this;
  }
  std::unique_ptr<HorizontalExtentImpl> clone() const { return cloneImpl(); }

  void merge(HorizontalExtentImpl const& other) { mergeImpl(other); }
  void merge(ast::HorizontalOffset const& other) { mergeImpl(other); }
  bool operator==(HorizontalExtentImpl const& other) const { return equalsImpl(other); }
  bool isPointwise() const { return isPointwiseImpl(); }
  std::unique_ptr<HorizontalExtentImpl> limit(int minus, int plus) const {
    return limitImpl(minus, plus);
  }

protected:
  virtual void addImpl(HorizontalExtentImpl const& other) = 0;
  virtual void mergeImpl(HorizontalExtentImpl const& other) = 0;
  virtual void mergeImpl(ast::HorizontalOffset const& other) = 0;
  virtual bool equalsImpl(HorizontalExtentImpl const& other) const = 0;
  virtual std::unique_ptr<HorizontalExtentImpl> cloneImpl() const = 0;
  virtual bool isPointwiseImpl() const = 0;
  virtual std::unique_ptr<HorizontalExtentImpl> limitImpl(int minus, int plus) const = 0;
};

class CartesianExtent : public HorizontalExtentImpl {
public:
  CartesianExtent(Extent const& iExtent, Extent const& jExtent) : extents_{iExtent, jExtent} {}
  CartesianExtent(int iMinus, int iPlus, int jMinus, int jPlus)
      : CartesianExtent(Extent(iMinus, iPlus), Extent(jMinus, jPlus)) {}

  CartesianExtent() : CartesianExtent(0, 0, 0, 0) {}

  int iMinus() const { return extents_[0].minus(); }
  int iPlus() const { return extents_[0].plus(); }
  int jMinus() const { return extents_[1].minus(); }
  int jPlus() const { return extents_[1].plus(); }

protected:
  void addImpl(HorizontalExtentImpl const& other) override {
    auto const& otherCartesian = dynamic_cast<CartesianExtent const&>(other);
    extents_[0] += otherCartesian.extents_[0];
    extents_[1] += otherCartesian.extents_[1];
  }

  void mergeImpl(HorizontalExtentImpl const& other) override {
    auto const& otherCartesian = dynamic_cast<CartesianExtent const&>(other);
    extents_[0].merge(otherCartesian.extents_[0]);
    extents_[1].merge(otherCartesian.extents_[1]);
  }

  void mergeImpl(ast::HorizontalOffset const& offset) override {
    auto const& offsetCartesian = ast::offset_cast<ast::CartesianOffset const&>(offset);
    extents_[0].merge(Extent(offsetCartesian.offsetI(), offsetCartesian.offsetI()));
    extents_[1].merge(Extent(offsetCartesian.offsetJ(), offsetCartesian.offsetJ()));
  }

  bool equalsImpl(HorizontalExtentImpl const& other) const override {
    auto const& otherCartesian = dynamic_cast<CartesianExtent const&>(other);
    return extents_[0] == otherCartesian.extents_[0] && extents_[1] == otherCartesian.extents_[1];
  }

  std::unique_ptr<HorizontalExtentImpl> cloneImpl() const override {
    return std::make_unique<CartesianExtent>(extents_[0].minus(), extents_[0].plus(),
                                             extents_[1].minus(), extents_[1].plus());
  }

  bool isPointwiseImpl() const override {
    return extents_[0].isPointwise() && extents_[1].isPointwise();
  }

  std::unique_ptr<HorizontalExtentImpl> limitImpl(int minus, int plus) const override {
    return std::make_unique<CartesianExtent>(extents_[0].limit(minus, plus),
                                             extents_[1].limit(minus, plus));
  }

private:
  std::array<Extent, 2> extents_;
};
class UnstructuredExtent : public HorizontalExtentImpl {
public:
  UnstructuredExtent(bool hasExtent) : hasExtent_{hasExtent} {}
  UnstructuredExtent() : UnstructuredExtent(false) {}

  bool hasExtent() const { return hasExtent_; }

protected:
  void addImpl(HorizontalExtentImpl const& other) override {
    auto const& otherUnstructured = dynamic_cast<UnstructuredExtent const&>(other);
    hasExtent_ = hasExtent_ || otherUnstructured.hasExtent_;
  }

  void mergeImpl(HorizontalExtentImpl const& other) override {
    auto const& otherUnstructured = dynamic_cast<UnstructuredExtent const&>(other);
    hasExtent_ = hasExtent_ || otherUnstructured.hasExtent_;
  }

  void mergeImpl(ast::HorizontalOffset const& offset) override {
    auto const& otherOffset = ast::offset_cast<ast::UnstructuredOffset const&>(offset);
    hasExtent_ = hasExtent_ || otherOffset.hasOffset();
  }

  bool equalsImpl(HorizontalExtentImpl const& other) const override {
    auto const& otherUnstructured = dynamic_cast<dawn::iir::UnstructuredExtent const&>(other);
    return hasExtent_ == otherUnstructured.hasExtent_;
  }

  std::unique_ptr<HorizontalExtentImpl> cloneImpl() const override {
    return std::make_unique<UnstructuredExtent>(hasExtent_);
  }

  bool isPointwiseImpl() const override { return !hasExtent_; }

  // TODO totally unclear what this should do...
  std::unique_ptr<HorizontalExtentImpl> limitImpl(int minus, int plus) const override {
    return clone();
  }

private:
  bool hasExtent_;
};

class HorizontalExtent {
public:
  // the default constructed horizontal extents creates a null-extent that can be compared to all
  // kind of grids
  HorizontalExtent() = default;

  HorizontalExtent(ast::HorizontalOffset const& hOffset) {
    *this = offset_dispatch(hOffset,
                            [](ast::CartesianOffset const& cOffset) {
                              return HorizontalExtent(ast::cartesian, cOffset.offsetI(),
                                                      cOffset.offsetI(), cOffset.offsetJ(),
                                                      cOffset.offsetJ());
                            },
                            [](ast::UnstructuredOffset const& uOffset) {
                              return HorizontalExtent(ast::unstructured, uOffset.hasOffset());
                            },
                            []() { return HorizontalExtent(); });
  }

  HorizontalExtent(ast::cartesian_) : impl_(std::make_unique<CartesianExtent>()) {}
  HorizontalExtent(ast::cartesian_, int iMinus, int iPlus, int jMinus, int jPlus)
      : impl_(std::make_unique<CartesianExtent>(iMinus, iPlus, jMinus, jPlus)) {}

  HorizontalExtent(ast::unstructured_) : impl_(std::make_unique<UnstructuredExtent>()) {}
  HorizontalExtent(ast::unstructured_, bool hasExtent)
      : impl_(std::make_unique<UnstructuredExtent>(hasExtent)) {}

  HorizontalExtent(HorizontalExtent const& other)
      : impl_(other.impl_ ? other.impl_->clone() : nullptr) {}
  HorizontalExtent(HorizontalExtent&& other) = default;
  HorizontalExtent& operator=(HorizontalExtent const& other) {
    if(other.impl_)
      impl_ = other.impl_->clone();
    else
      impl_ = nullptr;
    return *this;
  }
  HorizontalExtent& operator=(HorizontalExtent&& other) = default;

  HorizontalExtent(std::unique_ptr<HorizontalExtentImpl> impl) : impl_(std::move(impl)) {}

  template <typename T>
  friend T extent_cast(HorizontalExtent const&);
  template <typename CartFn, typename UnstructuredFn, typename ZeroFn>
  friend auto extent_dispatch(HorizontalExtent const& hExtent, CartFn const& cartFn,
                              UnstructuredFn const& unstructuredFn, ZeroFn const& zeroFn);

  bool operator==(HorizontalExtent const& other) const {
    if(impl_ && other.impl_)
      return *impl_ == *other.impl_;
    else if(impl_)
      return isPointwise();
    else if(other.impl_)
      return other.isPointwise();
    else
      return true;
  }
  bool operator!=(HorizontalExtent const& other) const { return !(*this == other); }
  HorizontalExtent& operator+=(HorizontalExtent const& other) {
    if(impl_ && other.impl_)
      *impl_ += *other.impl_;
    else if(other.impl_)
      *this = other;

    return *this;
  }
  void merge(const HorizontalExtent& other) {
    if(impl_ && other.impl_)
      impl_->merge(*other.impl_);
    else if(other.impl_)
      *this = other;
  }
  void merge(const ast::HorizontalOffset& other) {
    if(impl_)
      impl_->merge(other);
    else
      *this = other;
  }
  bool isPointwise() const { return !impl_ || impl_->isPointwise(); }
  HorizontalExtent limit(int minus, int plus) const {
    DAWN_ASSERT(minus <= 0 && plus >= 0);
    return impl_ ? impl_->limit(minus, plus) : *this;
  }

private:
  std::unique_ptr<HorizontalExtentImpl> impl_;
};

template <typename T>
T extent_cast(HorizontalExtent const& extent) {
  static std::remove_reference_t<T> nullExtent{};
  return extent.impl_ ? dynamic_cast<T>(*extent.impl_) : nullExtent;
}

/// @brief Three dimensional access extents of a field
/// @ingroup optimizer
class Extents {
public:
  enum class VerticalLoopOrderDir { VL_CounterLoopOrder, VL_InLoopOrder };

  /// @name Constructors and Assignment
  /// @{
  Extents();
  Extents(ast::Offsets const& offset);
  Extents(HorizontalExtent const& hExtent, Extent const& vExtent);

  Extents(ast::cartesian_, int extent1minus, int extent1plus, int extent2minus, int extent2plus,
          int extent3minus, int extent3plus);
  explicit Extents(ast::cartesian_);

  Extents(ast::unstructured_, bool hasOffset, Extent const& vExtent);
  explicit Extents(ast::unstructured_);
  /// @}

  bool hasVerticalCenter() const;

  /// @brief Limits the same extents, but limited to the interval [minus, plus]
  Extents limit(int minus, int plus) const;

  /// @brief Merge `this` with `other` and assign an Extents to `this` which is the union of the two
  ///
  /// @b Example:
  ///   If `this` is `{-1, 1, 0, 0, 0, 0}` and `other` is `{-2, 0, 0, 0, 1}` the result will be
  ///   `{-2, 1, 0, 0, 0, 1}`.
  void merge(const Extents& other);
  void merge(const ast::Offsets& offset);

  /// @brief resets vertical extants to {0, 0}
  void resetVerticalExtent();

  /// @brief Check if extent is pointwise in the horizontal direction, i.e. zero extent
  bool isHorizontalPointwise() const;

  /// @brief Check if extent is pointwise in the horizontal direction, i.e. zero extent
  bool isVerticalPointwise() const;

  /// @brief Check if extent is pointwise all directions, i.e. zero extent (which corresponds to the
  /// default initialized
  // extent)
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

std::ostream& operator<<(std::ostream& os, const Extents& extent);
std::string to_string(Extents const& extent);

template <typename CartFn, typename UnstructuredFn, typename ZeroFn>
auto extent_dispatch(HorizontalExtent const& hExtent, CartFn const& cartFn,
                     UnstructuredFn const& unstructuredFn, ZeroFn const& zeroFn) {
  if(hExtent.isPointwise())
    return zeroFn();

  HorizontalExtentImpl* ptr = hExtent.impl_.get();
  if(auto cartesianExtent = dynamic_cast<CartesianExtent const*>(ptr)) {
    return cartFn(*cartesianExtent);
  } else if(auto unstructuredExtent = dynamic_cast<UnstructuredExtent const*>(ptr)) {
    return unstructuredFn(*unstructuredExtent);
  } else {
    dawn_unreachable("unknown extent class");
  }
}

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
