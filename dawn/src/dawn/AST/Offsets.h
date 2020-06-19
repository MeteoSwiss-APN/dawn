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

#include "dawn/AST/GridType.h"
#include "dawn/AST/Tags.h"
#include "dawn/Support/Unreachable.h"

#include <array>
#include <memory>
#include <string>

namespace dawn {
namespace ast {

class Offsets;

static constexpr cartesian_ cartesian;
static constexpr unstructured_ unstructured;

class HorizontalOffsetImpl {
public:
  virtual ~HorizontalOffsetImpl() = default;
  std::unique_ptr<HorizontalOffsetImpl> clone() const;
  bool operator==(HorizontalOffsetImpl const& other) const;
  HorizontalOffsetImpl& operator+=(HorizontalOffsetImpl const& other);
  bool isZero() const;

protected:
  virtual std::unique_ptr<HorizontalOffsetImpl> cloneImpl() const = 0;
  virtual bool equalsImpl(HorizontalOffsetImpl const&) const = 0;
  virtual void addImpl(HorizontalOffsetImpl const&) = 0;
  virtual bool isZeroImpl() const = 0;
};

class CartesianOffset : public HorizontalOffsetImpl {
public:
  CartesianOffset() = default;
  CartesianOffset(int iOffset, int jOffset);
  explicit CartesianOffset(std::array<int, 2> const& offsets);

  int offsetI() const;
  int offsetJ() const;

protected:
  std::unique_ptr<HorizontalOffsetImpl> cloneImpl() const override;
  bool equalsImpl(HorizontalOffsetImpl const& other) const override;
  void addImpl(HorizontalOffsetImpl const& other) override;
  bool isZeroImpl() const override;

private:
  std::array<int, 2> horizontalOffset_;
};

class UnstructuredOffset : public HorizontalOffsetImpl {
public:
  UnstructuredOffset() = default;
  UnstructuredOffset(bool hasOffset);

  bool hasOffset() const;

protected:
  std::unique_ptr<HorizontalOffsetImpl> cloneImpl() const override;
  bool equalsImpl(HorizontalOffsetImpl const& other) const override;
  void addImpl(HorizontalOffsetImpl const& other) override;
  bool isZeroImpl() const override;

private:
  bool hasOffset_ = false;
};

class HorizontalOffset {
public:
  // the default constructed horizontal offset creates a null-offset that can be compared to all
  // kind of grids
  HorizontalOffset() = default;

  explicit HorizontalOffset(cartesian_);
  HorizontalOffset(cartesian_, int iOffset, int jOffset);

  HorizontalOffset(unstructured_);
  HorizontalOffset(unstructured_, bool hasOffset);

  HorizontalOffset(HorizontalOffset const& other);
  HorizontalOffset(HorizontalOffset&& other) = default;
  HorizontalOffset& operator=(HorizontalOffset const& other);
  HorizontalOffset& operator=(HorizontalOffset&& other) = default;

  bool operator==(HorizontalOffset const& other) const;
  bool operator!=(HorizontalOffset const& other) const;

  HorizontalOffset& operator+=(HorizontalOffset const& other);
  bool isZero() const;

  template <typename T>
  friend T offset_cast(HorizontalOffset const& offset);
  template <typename CartFn, typename UnstructuredFn, typename ZeroFn>
  friend auto offset_dispatch(HorizontalOffset const& hOffset, CartFn const& cartFn,
                              UnstructuredFn const& unstructuredFn, ZeroFn const& zeroFn);

  // impl may be a null ptr, i.e. HorizontalOffset may be in a uninitialized state and not
  // associated with a grid type
  bool hasType() const;
  GridType getGridType() const;

private:
  std::unique_ptr<HorizontalOffsetImpl> impl_;
};

/**
 * \brief casts offset to a given horizontal offset type. If the offset is a zero offset, an
 * appropriate zero offset of the given kind will be created.
 */
template <typename T>
T offset_cast(HorizontalOffset const& offset) {
  using PlainT = std::remove_reference_t<T>;
  static_assert(std::is_base_of_v<HorizontalOffsetImpl, PlainT>,
                "Can only be casted to a valid horizontal offset implementation");
  static_assert(std::is_const_v<PlainT>, "Can only be casted to const");
  static PlainT nullOffset{};
  return offset.impl_ ? dynamic_cast<T>(*offset.impl_) : nullOffset;
}
/**
 * \brief depending on the kind of horizontal offset, the appropriate function will be called
 *
 * Note, that you should only use this function if you cannot use `offset_cast` or the public
 * interface of HorizontalOffset.
 *
 * \param hOffset Horizontal offset on which we want to dispatch
 * \param cartFn Function to be called if hOffset is a cartesian offset. cartFn will be called with
 * hOffset casted to CartesianOffset
 * \param unstructuredFn Function to be called if hOffset is an
 * unstructured offset. unstructuredFn will be called with hOffset casted to UnstructuredOffset
 * \param zeroFn Function to be called if hOffset is a zero offset. zeroFn will be called with no
 * arguments
 */
template <typename CartFn, typename UnstructuredFn, typename ZeroFn>
auto offset_dispatch(HorizontalOffset const& hOffset, CartFn const& cartFn,
                     UnstructuredFn const& unstructuredFn, ZeroFn const& zeroFn) {
  if(hOffset.isZero())
    return zeroFn();

  HorizontalOffsetImpl* ptr = hOffset.impl_.get();
  if(auto cartesianOffset = dynamic_cast<CartesianOffset const*>(ptr)) {
    return cartFn(*cartesianOffset);
  } else if(auto unstructuredOffset = dynamic_cast<UnstructuredOffset const*>(ptr)) {
    return unstructuredFn(*unstructuredOffset);
  } else {
    dawn_unreachable("unknown offset class");
  }
}

class Offsets {
public:
  Offsets() = default;
  Offsets(HorizontalOffset const& hOffset, int vOffset);

  Offsets(cartesian_, int i, int j, int k);
  Offsets(cartesian_, std::array<int, 3> const& structuredOffsets);
  explicit Offsets(cartesian_);

  Offsets(unstructured_, bool hasOffset, int k);
  explicit Offsets(unstructured_);

  int verticalOffset() const;
  HorizontalOffset const& horizontalOffset() const;

  bool operator==(Offsets const& other) const;
  bool operator!=(Offsets const& other) const;
  Offsets& operator+=(Offsets const& other);

  bool isZero() const;

private:
  HorizontalOffset horizontalOffset_;
  int verticalOffset_ = 0;
};
Offsets operator+(Offsets o1, Offsets const& o2);

/**
 * For each component of :offset, calls `offset_to_string(name_of_offset, offset_value)`.
 * Concatenates all non-zero stringified offsets using :sep as a delimiter.
 */
template <typename F>
std::string to_string(cartesian_, Offsets const& offset, std::string const& sep,
                      F const& offset_to_string) {
  auto const& hoffset = offset_cast<CartesianOffset const&>(offset.horizontalOffset());
  auto const& voffset = offset.verticalOffset();
  std::string s;
  std::string csep = "";
  if(std::string ret = offset_to_string("i", hoffset.offsetI()); ret != "") {
    s += csep + ret;
    csep = sep;
  }
  if(std::string ret = offset_to_string("j", hoffset.offsetJ()); ret != "") {
    s += csep + ret;
    csep = sep;
  }
  if(std::string ret = offset_to_string("k", voffset); ret != "")
    s += csep + ret;
  return s;
}
std::string to_string(cartesian_, Offsets const& offset, std::string const& sep = ",");

std::string to_string(unstructured_, Offsets const& offset);

std::string to_string(Offsets const& offset);

} // namespace ast
} // namespace dawn
