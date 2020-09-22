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

#include "dawn/AST/Offsets.h"
#include "dawn/AST/ASTExpr.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Logger.h"
#include <memory>
#include <optional>

namespace dawn {
namespace ast {

// HorizontalOffsetImpl
std::unique_ptr<HorizontalOffsetImpl> HorizontalOffsetImpl::clone() const { return cloneImpl(); }

bool HorizontalOffsetImpl::operator==(HorizontalOffsetImpl const& other) const {
  return equalsImpl(other);
}

HorizontalOffsetImpl& HorizontalOffsetImpl::operator+=(HorizontalOffsetImpl const& other) {
  addImpl(other);
  return *this;
}

bool HorizontalOffsetImpl::isZero() const { return isZeroImpl(); }

// CartesianOffset
CartesianOffset::CartesianOffset(int iOffset, int jOffset) : horizontalOffset_{iOffset, jOffset} {}
CartesianOffset::CartesianOffset(std::array<int, 2> const& offsets) : horizontalOffset_(offsets) {}

int CartesianOffset::offsetI() const { return horizontalOffset_[0]; }
int CartesianOffset::offsetJ() const { return horizontalOffset_[1]; }

std::unique_ptr<HorizontalOffsetImpl> CartesianOffset::cloneImpl() const {
  return std::make_unique<CartesianOffset>(horizontalOffset_);
}
bool CartesianOffset::equalsImpl(HorizontalOffsetImpl const& other) const {
  auto const& co_other = dynamic_cast<CartesianOffset const&>(other);
  return co_other.horizontalOffset_ == horizontalOffset_;
}

void CartesianOffset::addImpl(HorizontalOffsetImpl const& other) {
  auto const& co_other = dynamic_cast<CartesianOffset const&>(other);
  horizontalOffset_[0] += co_other.horizontalOffset_[0];
  horizontalOffset_[1] += co_other.horizontalOffset_[1];
}

bool CartesianOffset::isZeroImpl() const {
  return horizontalOffset_[0] == 0 && horizontalOffset_[1] == 0;
}

// UnstructuredOffset
UnstructuredOffset::UnstructuredOffset(bool hasOffset) : hasOffset_(hasOffset) {}
bool UnstructuredOffset::hasOffset() const { return hasOffset_; }
std::unique_ptr<HorizontalOffsetImpl> UnstructuredOffset::cloneImpl() const {
  return std::make_unique<UnstructuredOffset>(hasOffset_);
}

bool UnstructuredOffset::equalsImpl(HorizontalOffsetImpl const& other) const {
  auto const& uo_other = dynamic_cast<UnstructuredOffset const&>(other);
  return uo_other.hasOffset_ == hasOffset_;
}
void UnstructuredOffset::addImpl(HorizontalOffsetImpl const& other) {
  auto const& uo_other = dynamic_cast<UnstructuredOffset const&>(other);
  hasOffset_ = hasOffset_ || uo_other.hasOffset_;
}

bool UnstructuredOffset::isZeroImpl() const { return !hasOffset_; }

// HorizontalOffset
HorizontalOffset::HorizontalOffset(cartesian_) : impl_(std::make_unique<CartesianOffset>()) {}
HorizontalOffset::HorizontalOffset(cartesian_, int iOffset, int jOffset)
    : impl_(std::make_unique<CartesianOffset>(iOffset, jOffset)) {}

HorizontalOffset::HorizontalOffset(unstructured_) : impl_(std::make_unique<UnstructuredOffset>()) {}
HorizontalOffset::HorizontalOffset(unstructured_, bool hasOffset)
    : impl_(std::make_unique<UnstructuredOffset>(hasOffset)) {}

HorizontalOffset::HorizontalOffset(HorizontalOffset const& other) { *this = other; }
HorizontalOffset& HorizontalOffset::operator=(HorizontalOffset const& other) {
  if(other.impl_)
    impl_ = other.impl_->clone();
  else
    impl_ = nullptr;
  return *this;
}

bool HorizontalOffset::operator==(HorizontalOffset const& other) const {
  if(impl_ && other.impl_)
    return *impl_ == *other.impl_;
  else if(impl_)
    return isZero();
  else if(other.impl_)
    return other.isZero();
  else
    return true;
}
bool HorizontalOffset::operator!=(HorizontalOffset const& other) const { return !(*this == other); }

HorizontalOffset& HorizontalOffset::operator+=(HorizontalOffset const& other) {
  if(impl_ && other.impl_)
    *impl_ += *other.impl_;
  else if(other.impl_)
    *this = other;
  return *this;
}
bool HorizontalOffset::isZero() const { return !impl_ || impl_->isZero(); }

bool HorizontalOffset::hasType() const { return impl_.get() != nullptr; }

GridType HorizontalOffset::getGridType() const {
  DAWN_ASSERT(hasType());
  if(dynamic_cast<CartesianOffset*>(impl_.get())) {
    return GridType::Cartesian;
  } else {
    return GridType::Unstructured;
  }
}

// Vertical Offset

bool VerticalOffset::operator==(VerticalOffset const& other) const {
  bool offsetEqual = verticalShift_ == other.verticalShift_;
  if(!verticalIndirection_ && !other.verticalIndirection_) {
    return offsetEqual;
  } else if(verticalIndirection_ && other.verticalIndirection_) {
    return offsetEqual && (*verticalIndirection_ == *other.verticalIndirection_);
  } else {
    return false;
  }
}

std::string VerticalOffset::getIndirectionFieldName() const {
  DAWN_ASSERT(hasIndirection());
  return std::dynamic_pointer_cast<FieldAccessExpr>(verticalIndirection_)->getName();
}
std::shared_ptr<const FieldAccessExpr> VerticalOffset::getIndirectionField() const {
  DAWN_ASSERT(hasIndirection());
  return std::dynamic_pointer_cast<const FieldAccessExpr>(verticalIndirection_);
}
void VerticalOffset::setIndirectionAccessID(int accessID) {
  DAWN_ASSERT(hasIndirection());
  auto field = std::dynamic_pointer_cast<FieldAccessExpr>(verticalIndirection_);
  auto data = field->getData<iir::IIRAccessExprData>();
  data.AccessID = std::make_optional(accessID);
}
std::optional<int> VerticalOffset::getIndirectionAccessID() const {
  DAWN_ASSERT(hasIndirection());
  auto field = std::dynamic_pointer_cast<FieldAccessExpr>(verticalIndirection_);
  auto data = field->getData<iir::IIRAccessExprData>();
  return data.AccessID;
}
std::shared_ptr<Expr>& VerticalOffset::getIndirectionFieldAsExpr() {
  DAWN_ASSERT(hasIndirection());
  return verticalIndirection_;
}

VerticalOffset VerticalOffset::operator+=(VerticalOffset const& other) {
  verticalShift_ += other.verticalShift_;
  if(other.verticalIndirection_ || verticalIndirection_) {
    DAWN_LOG(WARNING) << "operator += not well defined for vertical offsets with indirection";
    DAWN_ASSERT(false);
  }
  return *this;
}

VerticalOffset::VerticalOffset(int shift, const std::string& fieldName)
    : verticalShift_(shift), verticalIndirection_(std::make_shared<FieldAccessExpr>(fieldName)) {}

VerticalOffset::VerticalOffset(const VerticalOffset& other) { *this = other; }

VerticalOffset& VerticalOffset::operator=(VerticalOffset const& other) {
  verticalShift_ = other.verticalShift_;
  if(other.verticalIndirection_) {
    verticalIndirection_ = std::make_shared<FieldAccessExpr>(*other.getIndirectionField());
  } else {
    verticalIndirection_ = nullptr;
  }
  return *this;
}

// Offsets

Offsets::Offsets(HorizontalOffset const& hOffset, int vOffset)
    : horizontalOffset_(hOffset), verticalOffset_(VerticalOffset(vOffset)) {}
Offsets::Offsets(HorizontalOffset const& hOffset, int vOffset, const std::string& vIndirection)
    : horizontalOffset_(hOffset), verticalOffset_(VerticalOffset(vOffset, vIndirection)) {}

Offsets::Offsets(cartesian_, int i, int j, int k)
    : horizontalOffset_(cartesian, i, j), verticalOffset_(VerticalOffset(k)) {}
Offsets::Offsets(cartesian_, std::array<int, 3> const& structuredOffsets)
    : Offsets(cartesian, structuredOffsets[0], structuredOffsets[1], structuredOffsets[2]) {}
Offsets::Offsets(cartesian_, int i, int j, int k, const std::string& fieldName)
    : horizontalOffset_(cartesian, i, j), verticalOffset_(VerticalOffset(k, fieldName)) {}
Offsets::Offsets(cartesian_, std::array<int, 3> const& structuredOffsets,
                 const std::string& fieldName)
    : Offsets(cartesian, structuredOffsets[0], structuredOffsets[1], structuredOffsets[2],
              fieldName) {}
Offsets::Offsets(cartesian_) : horizontalOffset_(cartesian) {}

Offsets::Offsets(unstructured_, bool hasOffset, int k)
    : horizontalOffset_(unstructured, hasOffset), verticalOffset_(k) {}
Offsets::Offsets(unstructured_, bool hasOffset, int k, const std::string& fieldName)
    : horizontalOffset_(unstructured, hasOffset), verticalOffset_(k, fieldName) {}
Offsets::Offsets(unstructured_) : horizontalOffset_(unstructured) {}

std::string Offsets::getVerticalIndirectionFieldName() const {
  DAWN_ASSERT(hasVerticalIndirection());
  return verticalOffset_.getIndirectionFieldName();
}

std::shared_ptr<const FieldAccessExpr> Offsets::getVerticalIndirectionField() const {
  DAWN_ASSERT(hasVerticalIndirection());
  return verticalOffset_.getIndirectionField();
}
void Offsets::setVerticalIndirectionAccessID(int accessID) {
  DAWN_ASSERT(hasVerticalIndirection());
  verticalOffset_.setIndirectionAccessID(accessID);
}
std::optional<int> Offsets::getVerticalIndirectionAccessID() const {
  DAWN_ASSERT(hasVerticalIndirection());
  return verticalOffset_.getIndirectionAccessID();
}
std::shared_ptr<Expr>& Offsets::getVerticalIndirectionFieldAsExpr() {
  DAWN_ASSERT(hasVerticalIndirection());
  return verticalOffset_.getIndirectionFieldAsExpr();
}

void Offsets::setVerticalIndirection(const std::string& fieldName) {
  verticalOffset_ = VerticalOffset(verticalOffset_.getShift(), fieldName);
}

HorizontalOffset const& Offsets::horizontalOffset() const { return horizontalOffset_; }

bool Offsets::operator==(Offsets const& other) const {
  return horizontalOffset_ == other.horizontalOffset_ && verticalOffset_ == other.verticalOffset_;
}
bool Offsets::operator!=(Offsets const& other) const { return !(*this == other); }

Offsets& Offsets::operator+=(Offsets const& other) {
  horizontalOffset_ += other.horizontalOffset_;
  verticalOffset_ += other.verticalOffset_;
  return *this;
}

bool Offsets::isZero() const { return verticalOffset_ == 0 && horizontalOffset_.isZero(); }

std::string to_string(unstructured_, Offsets const& offset) {
  auto const& hoffset = offset_cast<UnstructuredOffset const&>(offset.horizontalOffset());
  auto const& voffset = offset.verticalShift();

  using namespace std::string_literals;
  return (hoffset.hasOffset() ? "<has_horizontal_offset>"s : "<no_horizontal_offset>"s) + "," +
         std::to_string(voffset);
}

std::string to_string(cartesian_, Offsets const& offsets, std::string const& sep) {
  return to_string(cartesian, offsets, sep,
                   [](std::string const&, int offset) { return std::to_string(offset); });
}

std::string to_string(Offsets const& offset) {
  return offset_dispatch(
      offset.horizontalOffset(),
      [&](CartesianOffset const&) { return to_string(cartesian, offset); },
      [&](UnstructuredOffset const&) { return to_string(unstructured, offset); },
      [&]() {
        using namespace std::string_literals;
        return "<no_horizontal_offset>,"s + std::to_string(offset.verticalShift());
      });
}

Offsets operator+(Offsets o1, Offsets const& o2) { return o1 += o2; }

} // namespace ast
} // namespace dawn
