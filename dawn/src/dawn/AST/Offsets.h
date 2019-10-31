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
//
#ifndef DAWN_AST_OFFSET_H
#define DAWN_AST_OFFSET_H

#include "dawn/Support/Unreachable.h"

#include <memory>
#include <vector>

namespace dawn::iir {
class HorizontalExtent;
}
namespace dawn::ast {

class Offsets;

struct cartesian_ {};
static constexpr cartesian_ cartesian;

struct unstructured_ {};
static constexpr unstructured_ unstructured;

class HorizontalOffsetImpl {
public:
  virtual ~HorizontalOffsetImpl() = default;
  std::unique_ptr<HorizontalOffsetImpl> clone() const { return cloneImpl(); }

  bool operator==(HorizontalOffsetImpl const& other) const { return equalsImpl(other); }

  HorizontalOffsetImpl& operator+=(HorizontalOffsetImpl const& other) {
    addImpl(other);
    return *this;
  }

  bool isZero() const { return isZeroImpl(); }

protected:
  virtual std::unique_ptr<HorizontalOffsetImpl> cloneImpl() const = 0;
  virtual bool equalsImpl(HorizontalOffsetImpl const&) const = 0;
  virtual void addImpl(HorizontalOffsetImpl const&) = 0;
  virtual bool isZeroImpl() const = 0;
};

class CartesianOffset : public HorizontalOffsetImpl {
public:
  CartesianOffset(int iOffset, int jOffset) : horizontalOffset_{iOffset, jOffset} {}
  explicit CartesianOffset(std::array<int, 2> const& offsets) : horizontalOffset_(offsets) {}
  CartesianOffset() = default;

  int offsetI() const { return horizontalOffset_[0]; }
  int offsetJ() const { return horizontalOffset_[1]; }

protected:
  std::unique_ptr<HorizontalOffsetImpl> cloneImpl() const override {
    return std::make_unique<CartesianOffset>(horizontalOffset_);
  }
  bool equalsImpl(HorizontalOffsetImpl const& other) const override;
  void addImpl(HorizontalOffsetImpl const& other) override;
  bool isZeroImpl() const override {
    return horizontalOffset_[0] == 0 && horizontalOffset_[1] == 0;
  }

private:
  std::array<int, 2> horizontalOffset_;
};

class UnstructuredOffset : public HorizontalOffsetImpl {
public:
  UnstructuredOffset() = default;
  UnstructuredOffset(bool hasOffset) : hasOffset_(hasOffset) {}

  bool hasOffset() const { return hasOffset_; }

protected:
  std::unique_ptr<HorizontalOffsetImpl> cloneImpl() const override {
    return std::make_unique<UnstructuredOffset>(hasOffset_);
  }
  bool equalsImpl(HorizontalOffsetImpl const& other) const override;
  void addImpl(HorizontalOffsetImpl const& other) override;
  bool isZeroImpl() const override { return !hasOffset_; }

private:
  bool hasOffset_ = false;
};

class HorizontalOffset {
public:
  // the default constructed horizontal offset creates a null-offset that can be compared to all
  // kind of grids
  HorizontalOffset() = default;

  explicit HorizontalOffset(cartesian_) : impl_(std::make_unique<CartesianOffset>()) {}
  HorizontalOffset(cartesian_, int iOffset, int jOffset)
      : impl_(std::make_unique<CartesianOffset>(iOffset, jOffset)) {}

  HorizontalOffset(unstructured_) : impl_(std::make_unique<UnstructuredOffset>()) {}
  HorizontalOffset(unstructured_, bool hasOffset)
      : impl_(std::make_unique<UnstructuredOffset>(hasOffset)) {}

  HorizontalOffset(HorizontalOffset const& other) { *this = other; }
  HorizontalOffset& operator=(HorizontalOffset const& other) {
    if(other.impl_)
      impl_ = other.impl_->clone();
    else
      impl_ = nullptr;
    return *this;
  }
  HorizontalOffset(HorizontalOffset&& other) = default;
  HorizontalOffset& operator=(HorizontalOffset&& other) = default;

  bool operator==(HorizontalOffset const& other) const {
    if(impl_ && other.impl_)
      return *impl_ == *other.impl_;
    else if(impl_)
      return isZero();
    else if(other.impl_)
      return other.isZero();
    else
      return true;
  }
  bool operator!=(HorizontalOffset const& other) const { return !(*this == other); }

  HorizontalOffset& operator+=(HorizontalOffset const& other) {
    if(impl_ && other.impl_)
      *impl_ += *other.impl_;
    else if(other.impl_)
      *this = other;
    return *this;
  }
  bool isZero() const { return !impl_ || impl_->isZero(); }

  template <typename T>
  friend T offset_cast(HorizontalOffset const& offset);
  friend std::string to_string(Offsets const& offset);
  template <typename CartFn, typename UnstructuredFn, typename ZeroFn>
  friend auto offset_dispatch(HorizontalOffset const& hOffset, CartFn const& cartFn,
                              UnstructuredFn const& unstructuredFn, ZeroFn const& zeroFn);

private:
  std::unique_ptr<HorizontalOffsetImpl> impl_;
};

template <typename T>
T offset_cast(HorizontalOffset const& offset) {
  using PlainT = std::remove_reference_t<T>;
  static_assert(std::is_base_of_v<HorizontalOffsetImpl, PlainT>,
                "Can only be casted to a valid horizontal offset implementation");
  static_assert(std::is_const_v<PlainT>, "Can only be casted to const");
  static PlainT nullOffset{};
  return offset.impl_ ? dynamic_cast<T>(*offset.impl_) : nullOffset;
}

class Offsets {
public:
  Offsets() = default;
  Offsets(HorizontalOffset const& hOffset, int vOffset)
      : horizontalOffset_(hOffset), verticalOffset_(vOffset) {}

  Offsets(cartesian_, int i, int j, int k)
      : horizontalOffset_(cartesian, i, j), verticalOffset_(k) {}
  Offsets(cartesian_, std::array<int, 3> const& structuredOffsets)
      : Offsets(cartesian, structuredOffsets[0], structuredOffsets[1], structuredOffsets[2]) {}
  explicit Offsets(cartesian_) : horizontalOffset_(cartesian) {}

  Offsets(unstructured_, bool hasOffset, int k)
      : horizontalOffset_(unstructured, hasOffset), verticalOffset_(k) {}
  explicit Offsets(unstructured_) : horizontalOffset_(unstructured) {}
  int verticalOffset() const { return verticalOffset_; }
  HorizontalOffset const& horizontalOffset() const { return horizontalOffset_; }

  bool operator==(Offsets const& other) const {
    return horizontalOffset_ == other.horizontalOffset_ && verticalOffset_ == other.verticalOffset_;
  }
  bool operator!=(Offsets const& other) const { return !(*this == other); }

  Offsets& operator+=(Offsets const& other) {
    horizontalOffset_ += other.horizontalOffset_;
    verticalOffset_ += other.verticalOffset_;
    return *this;
  }

  // corresponds to the default initialized offset
  bool isZero() const { return verticalOffset_ == 0 && horizontalOffset_.isZero(); }

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

} // namespace dawn::ast
#endif
