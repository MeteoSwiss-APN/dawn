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

namespace dawn::ast {

struct structured_ {};
static constexpr structured_ structured;

class HorizontalOffsetImpl {
public:
  virtual ~HorizontalOffsetImpl() = default;
  std::unique_ptr<HorizontalOffsetImpl> clone() const {
    return std::unique_ptr<HorizontalOffsetImpl>(cloneImpl());
  }

  bool operator==(HorizontalOffsetImpl const& other) const { return equalsImpl(other); }

  HorizontalOffsetImpl& operator+=(HorizontalOffsetImpl const& other) { return addImpl(other); }

  bool isZero() const { return isZeroImpl(); }

protected:
  virtual HorizontalOffsetImpl* cloneImpl() const = 0;
  virtual bool equalsImpl(HorizontalOffsetImpl const&) const = 0;
  virtual HorizontalOffsetImpl& addImpl(HorizontalOffsetImpl const&) = 0;
  virtual bool isZeroImpl() const = 0;
};

class StructuredOffset : public HorizontalOffsetImpl {
public:
  StructuredOffset(int iOffset, int jOffset) : horizontalOffset_({iOffset, jOffset}) {}
  StructuredOffset(std::array<int, 2> const& offsets) : horizontalOffset_(offsets) {}
  StructuredOffset() = default;

  int offsetI() const { return horizontalOffset_[0]; }
  int offsetJ() const { return horizontalOffset_[1]; }

protected:
  HorizontalOffsetImpl* cloneImpl() const override {
    return new StructuredOffset{horizontalOffset_};
  }
  bool equalsImpl(HorizontalOffsetImpl const& other) const override {
    auto const& so_other = dynamic_cast<StructuredOffset const&>(other);
    return so_other.horizontalOffset_ == horizontalOffset_;
  }
  StructuredOffset& addImpl(HorizontalOffsetImpl const& other) override {
    auto const& so_other = dynamic_cast<StructuredOffset const&>(other);
    horizontalOffset_[0] += so_other.horizontalOffset_[0];
    horizontalOffset_[1] += so_other.horizontalOffset_[1];
    return *this;
  }
  bool isZeroImpl() const override {
    return horizontalOffset_[0] == 0 && horizontalOffset_[1] == 0;
  }

private:
  std::array<int, 2> horizontalOffset_;
};

class HorizontalOffset {
public:
  HorizontalOffset(structured_) : impl_(std::make_unique<StructuredOffset>()) {}
  HorizontalOffset(structured_, int iOffset, int jOffset)
      : impl_(std::make_unique<StructuredOffset>(iOffset, jOffset)) {}
  HorizontalOffset(HorizontalOffset const& other) : impl_(other.impl_->clone()) {}
  HorizontalOffset& operator=(HorizontalOffset const& other) {
    impl_ = other.impl_->clone();
    return *this;
  }
  HorizontalOffset(HorizontalOffset&& other) = default;
  HorizontalOffset& operator=(HorizontalOffset&& other) = default;

  bool operator==(HorizontalOffset const& other) const { return *impl_ == *other.impl_; }
  bool operator!=(HorizontalOffset const& other) const { return !(*this == other); }

  HorizontalOffset& operator+=(HorizontalOffset const& other) {
    *impl_ += *other.impl_;
    return *this;
  }
  bool isZero() const { return impl_->isZero(); }

  template <typename T>
  friend T offset_cast(HorizontalOffset const& offset);

private:
  std::unique_ptr<HorizontalOffsetImpl> impl_;
};

template <typename T>
T offset_cast(HorizontalOffset const& offset) {
  return dynamic_cast<T>(*offset.impl_);
}

class Offsets {
public:
  Offsets(structured_, int i, int j, int k)
      : horizontalOffset_(structured, i, j), verticalOffset_(k) {}
  Offsets(structured_, std::array<int, 3> const& structuredOffsets)
      : Offsets(structured, structuredOffsets[0], structuredOffsets[1], structuredOffsets[2]) {}
  Offsets(structured_) : horizontalOffset_(structured) {}

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

  bool isZero() const { return verticalOffset_ == 0 && horizontalOffset_.isZero(); }

private:
  HorizontalOffset horizontalOffset_;
  int verticalOffset_ = 0;
};
template <typename F>
std::string toString(Offsets const& offset, std::string const& sep, F&& f) {
  auto const& hoffset = ast::offset_cast<StructuredOffset const&>(offset.horizontalOffset());
  auto const& voffset = offset.verticalOffset();
  std::string s;
  if(auto ret = f("i", hoffset.offsetI()); ret != "")
    s += ret + sep;
  if(auto ret = f("j", hoffset.offsetJ()); ret != "")
    s += ret + sep;
  if(auto ret = f("k", voffset); ret != "")
    s += ret + sep;
  return s;
}
std::string toString(Offsets const& offset, std::string const& sep = ",");
// the default printer prints "i, j, k" for structured grids, otherwise you should use toString
std::ostream& operator<<(std::ostream& os, Offsets const& offsets);

} // namespace dawn::ast
#endif

