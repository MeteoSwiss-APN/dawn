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

class Dimension {
public:
  explicit constexpr Dimension(std::string_view name) : name_(name) {}
  std::string_view const& getName() const { return name_; };
  bool operator==(Dimension const& other) const { return name_ == other.name_; }

private:
  std::string_view name_;
};

class HorizontalOffsetImpl {
public:
  virtual ~HorizontalOffsetImpl() = default;
  std::unique_ptr<HorizontalOffsetImpl> clone() const {
    return std::unique_ptr<HorizontalOffsetImpl>(cloneImpl());
  }

  bool operator==(HorizontalOffsetImpl const& other) const { return equalsImpl(other); }

  HorizontalOffsetImpl& operator+=(HorizontalOffsetImpl const& other) { return addImpl(other); }

  std::vector<std::reference_wrapper<const Dimension>> getDimensions() const {
    return getDimensionsImpl();
  }
  int getOffset(Dimension const& dim) const { return getOffsetImpl(dim); }

protected:
  virtual HorizontalOffsetImpl* cloneImpl() const = 0;
  virtual bool equalsImpl(HorizontalOffsetImpl const&) const = 0;
  virtual HorizontalOffsetImpl& addImpl(HorizontalOffsetImpl const&) = 0;
  virtual std::vector<std::reference_wrapper<const Dimension>> getDimensionsImpl() const = 0;
  virtual int getOffsetImpl(Dimension const& dim) const = 0;
};

class StructuredOffset : public HorizontalOffsetImpl {
public:
  StructuredOffset(std::array<int, 2> const& offset) : horizontalOffset_(offset) {}
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
  std::vector<std::reference_wrapper<const Dimension>> getDimensionsImpl() const override {
    return {std::cref(dimensionI_), std::cref(dimensionJ_)};
  }

  int getOffsetImpl(Dimension const& dim) const {
    if(dim == dimensionI_)
      return horizontalOffset_[0];
    else
      return horizontalOffset_[1];
  }

private:
  std::array<int, 2> horizontalOffset_;

  static constexpr Dimension dimensionI_{"i"};
  static constexpr Dimension dimensionJ_{"j"};
};

class HorizontalOffset {
public:
  HorizontalOffset(structured_) : impl_(std::make_unique<StructuredOffset>()) {}
  HorizontalOffset(structured_, std::array<int, 2> const& offset)
      : impl_(std::make_unique<StructuredOffset>(offset)) {}
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
  std::vector<std::reference_wrapper<const Dimension>> getDimensions() const {
    return impl_->getDimensions();
  }
  int getOffset(Dimension const& dim) const { return impl_->getOffset(dim); }

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
  Offsets(structured_, std::array<int, 3> const& structuredOffsets)
      : horizontalOffset_(structured,
                          std::array<int, 2>{{structuredOffsets[0], structuredOffsets[1]}}),
        verticalOffset_(structuredOffsets[2]) {}
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

  // TODO this I guess weshould remove
  int getOffset(int i) const {
    auto const& hoffset = ast::offset_cast<StructuredOffset const&>(horizontalOffset());
    auto const& voffset = verticalOffset();
    switch(i) {
    case 0:
      return hoffset.offsetI();
    case 1:
      return hoffset.offsetJ();
    case 2:
      return voffset;
    default:
      dawn_unreachable("invalid offset");
    }
  }

  int getOffset(Dimension const& dim) const {
    if(dim == verticalDimension_)
      return verticalOffset_;
    return horizontalOffset().getOffset(dim);
  }

  std::vector<std::reference_wrapper<Dimension const>> getDimensions() const {
    std::vector<std::reference_wrapper<Dimension const>> ret = horizontalOffset().getDimensions();
    ret.push_back(std::cref(verticalDimension_));
    return ret;
  }

private:
  HorizontalOffset horizontalOffset_;
  int verticalOffset_ = 0;

  static constexpr Dimension verticalDimension_{"k"};
};
std::ostream& operator<<(std::ostream& os, Offsets const& offsets);

} // namespace dawn::ast
#endif

