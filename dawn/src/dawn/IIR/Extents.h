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

#include "LoopOrder.h"

#include "dawn/AST/GridType.h"
#include "dawn/AST/Offsets.h"

#include <array>
#include <iosfwd>
#include <optional>

namespace dawn {
namespace iir {

class Extents;

/// @brief Access extent of a single dimension
/// @ingroup optimizer
class Extent {

public:
  /// @name Constructors and Assignment
  /// @{
  Extent(int minus, int plus);
  explicit Extent(int offset);
  Extent();
  /// @}

  int minus() const { return minus_; }
  int plus() const { return plus_; }

  /// @name Operators
  /// @{
  void merge(const Extent& other);
  void merge(int other);

  void limit(Extent const& other);

  Extent& operator+=(const Extent& other);

  bool operator==(const Extent& other) const;
  bool operator!=(const Extent& other) const;

  bool isPointwise() const;
  /// @}
  //

private:
  int minus_;
  int plus_;
};
Extent operator+(Extent lhs, Extent const& rhs);
Extent merge(Extent lhs, Extent const& rhs);
Extent limit(Extent lhs, Extent const& rhs);

class HorizontalExtentImpl {
public:
  HorizontalExtentImpl() = default;
  virtual ~HorizontalExtentImpl() = default;

  HorizontalExtentImpl& operator+=(HorizontalExtentImpl const& other);
  std::unique_ptr<HorizontalExtentImpl> clone() const;

  void merge(HorizontalExtentImpl const& other);
  void addCenter();
  bool operator==(HorizontalExtentImpl const& other) const;
  bool isPointwise() const;
  void limit(HorizontalExtentImpl const& other);

protected:
  virtual void addImpl(HorizontalExtentImpl const& other) = 0;
  virtual void mergeImpl(HorizontalExtentImpl const& other) = 0;
  virtual void addCenterImpl() = 0;
  virtual bool equalsImpl(HorizontalExtentImpl const& other) const = 0;
  virtual std::unique_ptr<HorizontalExtentImpl> cloneImpl() const = 0;
  virtual bool isPointwiseImpl() const = 0;
  virtual void limitImpl(HorizontalExtentImpl const& other) = 0;
};

class CartesianExtent : public HorizontalExtentImpl {
public:
  CartesianExtent(Extent const& iExtent, Extent const& jExtent);
  CartesianExtent(int iMinus, int iPlus, int jMinus, int jPlus);

  CartesianExtent();

  int iMinus() const;
  int iPlus() const;
  int jMinus() const;
  int jPlus() const;

protected:
  void addImpl(HorizontalExtentImpl const& other) override;

  void mergeImpl(HorizontalExtentImpl const& other) override;
  void addCenterImpl() override;
  bool equalsImpl(HorizontalExtentImpl const& other) const override;
  std::unique_ptr<HorizontalExtentImpl> cloneImpl() const override;

  bool isPointwiseImpl() const override;
  void limitImpl(HorizontalExtentImpl const& other) override;

private:
  std::array<Extent, 2> extents_;
};

class UnstructuredExtent : public HorizontalExtentImpl {
public:
  UnstructuredExtent(bool hasExtent);
  UnstructuredExtent();

  bool hasExtent() const;

protected:
  void addImpl(HorizontalExtentImpl const& other) override;
  void mergeImpl(HorizontalExtentImpl const& other) override;
  void addCenterImpl() override;
  bool equalsImpl(HorizontalExtentImpl const& other) const override;
  std::unique_ptr<HorizontalExtentImpl> cloneImpl() const override;
  bool isPointwiseImpl() const override;
  void limitImpl(HorizontalExtentImpl const& other) override;

private:
  bool hasExtent_;
};

class HorizontalExtent {
public:
  // the default constructed horizontal extents creates a null-extent that can be compared to all
  // kind of grids
  HorizontalExtent() = default;

  HorizontalExtent(ast::HorizontalOffset const& hOffset);

  HorizontalExtent(ast::cartesian_);
  HorizontalExtent(ast::cartesian_, int iMinus, int iPlus, int jMinus, int jPlus);

  HorizontalExtent(ast::unstructured_);
  HorizontalExtent(ast::unstructured_, bool hasExtent);

  HorizontalExtent(HorizontalExtent const& other);
  HorizontalExtent(HorizontalExtent&& other) = default;
  HorizontalExtent& operator=(HorizontalExtent const& other);
  HorizontalExtent& operator=(HorizontalExtent&& other) = default;

  HorizontalExtent(std::unique_ptr<HorizontalExtentImpl> impl);

  template <typename T>
  friend T extent_cast(HorizontalExtent const&);
  template <typename CartFn, typename UnstructuredFn, typename ZeroFn>
  friend auto extent_dispatch(HorizontalExtent const& hExtent, CartFn const& cartFn,
                              UnstructuredFn const& unstructuredFn, ZeroFn const& zeroFn);

  bool operator==(HorizontalExtent const& other) const;
  bool operator!=(HorizontalExtent const& other) const;
  HorizontalExtent& operator+=(HorizontalExtent const& other);
  void merge(HorizontalExtent const& other);
  void merge(ast::HorizontalOffset const& other);
  bool isPointwise() const;
  void limit(HorizontalExtent const& other);

  bool hasType() const;
  ast::GridType getType() const;

private:
  std::unique_ptr<HorizontalExtentImpl> impl_;
};

/**
 * \brief casts extent to a given horizontal extent type. If the extent is a zero extent, an
 * appropriate zero extent of the given kind will be created.
 */
template <typename T>
T extent_cast(HorizontalExtent const& extent) {
  using PlainT = std::remove_reference_t<T>;
  static_assert(std::is_base_of_v<HorizontalExtentImpl, PlainT>,
                "Can only be cast to a valid horizontal extent implementation");
  static_assert(std::is_const_v<PlainT>, "Can only be cast to const");
  static PlainT nullExtent{};
  return extent.impl_ ? dynamic_cast<T>(*extent.impl_) : nullExtent;
}

/**
 * \brief depending on the kind of horizontal extent, the appropriate function will be called
 *
 * Note, that you should only use this function if you cannot use `extent_cast` or the public
 * interface of HorizontalExtent.
 *
 * \param hExtent Horizontal extent on which we want to dispatch
 * \param cartFn Function to be called if hExtent is a cartesian extent. cartFn will be called with
 * hExtent casted to CartesianExtent
 * \param unstructuredFn Function to be called if hExtent is an
 * unstructured extent. unstructuredFn will be called with hExtent casted to UnstructuredExtent
 * \param zeroFn Function to be called if hExtent is a zero extent. zeroFn will be called with no
 * arguments
 */
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

/// @brief Three dimensional access extents of a field
/// @ingroup optimizer
class Extents {
public:
  enum class VerticalLoopOrderDir { CounterLoopOrder, InLoopOrder };

  /// @name Constructors and Assignment
  /// @{
  Extents();
  explicit Extents(ast::Offsets const& offset);
  Extents(HorizontalExtent const& hExtent, Extent const& vExtent);

  Extents(ast::cartesian_, int extent1minus, int extent1plus, int extent2minus, int extent2plus,
          int extent3minus, int extent3plus);
  explicit Extents(ast::cartesian_);

  Extents(ast::unstructured_, bool hasExtent, Extent const& vExtent);
  explicit Extents(ast::unstructured_);
  /// @}

  bool hasVerticalCenter() const;

  /// @brief Limits the same extents, but limited to the extent given by other
  void limit(Extents const& other);

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
Extents merge(Extents lhs, Extents const& rhs);
Extents limit(Extents lhs, Extents const& rhs);

std::ostream& operator<<(std::ostream& os, const Extents& extent);
std::string to_string(Extents const& extent);

} // namespace iir
} // namespace dawn

namespace std {

template <>
struct hash<dawn::iir::Extents> {
  size_t operator()(const dawn::iir::Extents& extent) const;
};

} // namespace std
