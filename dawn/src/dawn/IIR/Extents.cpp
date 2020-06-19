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

#include "dawn/IIR/Extents.h"
#include "dawn/AST/GridType.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/HashCombine.h"
#include "dawn/Support/StringUtil.h"
#include "dawn/Support/Unreachable.h"

#include <algorithm>

namespace dawn::iir {

// Extent
Extent::Extent(int minus, int plus) : minus_(minus), plus_(plus) { DAWN_ASSERT(minus <= plus); }
Extent::Extent(int offset) : Extent(offset, offset) {}
Extent::Extent() : Extent(0, 0) {}
void Extent::merge(const Extent& other) {
  minus_ = std::min(minus_, other.minus_);
  plus_ = std::max(plus_, other.plus_);
}
void Extent::merge(int other) { merge(Extent{other, other}); }
void Extent::limit(Extent const& other) {
  minus_ = std::max(minus_, other.minus_);
  plus_ = std::min(plus_, other.plus_);
}
Extent& Extent::operator+=(const Extent& other) {
  minus_ += other.minus_;
  plus_ += other.plus_;
  return *this;
}
bool Extent::operator==(const Extent& other) const {
  return minus_ == other.minus_ && plus_ == other.plus_;
}
bool Extent::operator!=(const Extent& other) const { return !(*this == other); }
bool Extent::isPointwise() const { return plus_ == 0 && minus_ == 0; }

Extent operator+(Extent lhs, Extent const& rhs) { return lhs += rhs; }
Extent merge(Extent lhs, Extent const& rhs) {
  lhs.merge(rhs);
  return lhs;
}
Extent limit(Extent lhs, Extent const& rhs) {
  lhs.limit(rhs);
  return rhs;
}

// HorizontalExtentImpl
HorizontalExtentImpl& HorizontalExtentImpl::operator+=(HorizontalExtentImpl const& other) {
  addImpl(other);
  return *this;
}
std::unique_ptr<HorizontalExtentImpl> HorizontalExtentImpl::clone() const { return cloneImpl(); }
void HorizontalExtentImpl::merge(HorizontalExtentImpl const& other) { mergeImpl(other); }
void HorizontalExtentImpl::addCenter() { addCenterImpl(); }
bool HorizontalExtentImpl::operator==(HorizontalExtentImpl const& other) const {
  return equalsImpl(other);
}
bool HorizontalExtentImpl::isPointwise() const { return isPointwiseImpl(); }
void HorizontalExtentImpl::limit(HorizontalExtentImpl const& other) { limitImpl(other); }

// CartesianExtent
CartesianExtent::CartesianExtent(Extent const& iExtent, Extent const& jExtent)
    : extents_{iExtent, jExtent} {}
CartesianExtent::CartesianExtent(int iMinus, int iPlus, int jMinus, int jPlus)
    : CartesianExtent(Extent(iMinus, iPlus), Extent(jMinus, jPlus)) {}
CartesianExtent::CartesianExtent() : CartesianExtent(0, 0, 0, 0) {}
int CartesianExtent::iMinus() const { return extents_[0].minus(); }
int CartesianExtent::iPlus() const { return extents_[0].plus(); }
int CartesianExtent::jMinus() const { return extents_[1].minus(); }
int CartesianExtent::jPlus() const { return extents_[1].plus(); }

void CartesianExtent::addImpl(HorizontalExtentImpl const& other) {
  auto const& otherCartesian = dynamic_cast<CartesianExtent const&>(other);
  extents_[0] += otherCartesian.extents_[0];
  extents_[1] += otherCartesian.extents_[1];
}
void CartesianExtent::mergeImpl(HorizontalExtentImpl const& other) {
  auto const& otherCartesian = dynamic_cast<CartesianExtent const&>(other);
  extents_[0].merge(otherCartesian.extents_[0]);
  extents_[1].merge(otherCartesian.extents_[1]);
}

void CartesianExtent::addCenterImpl() { mergeImpl(CartesianExtent()); }

bool CartesianExtent::equalsImpl(HorizontalExtentImpl const& other) const {
  auto const& otherCartesian = dynamic_cast<CartesianExtent const&>(other);
  return extents_[0] == otherCartesian.extents_[0] && extents_[1] == otherCartesian.extents_[1];
}

std::unique_ptr<HorizontalExtentImpl> CartesianExtent::cloneImpl() const {
  return std::make_unique<CartesianExtent>(extents_[0].minus(), extents_[0].plus(),
                                           extents_[1].minus(), extents_[1].plus());
}
bool CartesianExtent::isPointwiseImpl() const {
  return extents_[0].isPointwise() && extents_[1].isPointwise();
}

void CartesianExtent::limitImpl(HorizontalExtentImpl const& other) {
  auto const& otherCartesian = dynamic_cast<CartesianExtent const&>(other);
  extents_[0].limit(otherCartesian.extents_[0]);
  extents_[1].limit(otherCartesian.extents_[1]);
}

// UnstructuredExtent
UnstructuredExtent::UnstructuredExtent(bool hasExtent) : hasExtent_{hasExtent} {}
UnstructuredExtent::UnstructuredExtent() : UnstructuredExtent(false) {}

bool UnstructuredExtent::hasExtent() const { return hasExtent_; }
void UnstructuredExtent::addImpl(HorizontalExtentImpl const& other) {
  auto const& otherUnstructured = dynamic_cast<UnstructuredExtent const&>(other);
  hasExtent_ = hasExtent_ || otherUnstructured.hasExtent_;
}

void UnstructuredExtent::mergeImpl(HorizontalExtentImpl const& other) {
  auto const& otherUnstructured = dynamic_cast<UnstructuredExtent const&>(other);
  hasExtent_ = hasExtent_ || otherUnstructured.hasExtent_;
}

void UnstructuredExtent::addCenterImpl() { mergeImpl(UnstructuredExtent()); }

bool UnstructuredExtent::equalsImpl(HorizontalExtentImpl const& other) const {
  auto const& otherUnstructured = dynamic_cast<dawn::iir::UnstructuredExtent const&>(other);
  return hasExtent_ == otherUnstructured.hasExtent_;
}

std::unique_ptr<HorizontalExtentImpl> UnstructuredExtent::cloneImpl() const {
  return std::make_unique<UnstructuredExtent>(hasExtent_);
}

bool UnstructuredExtent::isPointwiseImpl() const { return !hasExtent_; }

void UnstructuredExtent::limitImpl(HorizontalExtentImpl const& other) {
  auto const& otherUnstructured = dynamic_cast<dawn::iir::UnstructuredExtent const&>(other);
  hasExtent_ = hasExtent_ && otherUnstructured.hasExtent_;
}

// HorizontalExtent
HorizontalExtent::HorizontalExtent(ast::HorizontalOffset const& hOffset) {
  *this = offset_dispatch(
      hOffset,
      [](ast::CartesianOffset const& cOffset) {
        return HorizontalExtent(ast::cartesian, cOffset.offsetI(), cOffset.offsetI(),
                                cOffset.offsetJ(), cOffset.offsetJ());
      },
      [](ast::UnstructuredOffset const& uOffset) {
        return HorizontalExtent(ast::unstructured, uOffset.hasOffset());
      },
      []() { return HorizontalExtent(); });
}
HorizontalExtent::HorizontalExtent(ast::cartesian_) : impl_(std::make_unique<CartesianExtent>()) {}
HorizontalExtent::HorizontalExtent(ast::cartesian_, int iMinus, int iPlus, int jMinus, int jPlus)
    : impl_(std::make_unique<CartesianExtent>(iMinus, iPlus, jMinus, jPlus)) {}

HorizontalExtent::HorizontalExtent(ast::unstructured_)
    : impl_(std::make_unique<UnstructuredExtent>()) {}
HorizontalExtent::HorizontalExtent(ast::unstructured_, bool hasExtent)
    : impl_(std::make_unique<UnstructuredExtent>(hasExtent)) {}

HorizontalExtent::HorizontalExtent(HorizontalExtent const& other)
    : impl_(other.impl_ ? other.impl_->clone() : nullptr) {}
HorizontalExtent& HorizontalExtent::operator=(HorizontalExtent const& other) {
  if(other.impl_)
    impl_ = other.impl_->clone();
  else
    impl_ = nullptr;
  return *this;
}

HorizontalExtent::HorizontalExtent(std::unique_ptr<HorizontalExtentImpl> impl)
    : impl_(std::move(impl)) {}
bool HorizontalExtent::operator==(HorizontalExtent const& other) const {
  if(impl_ && other.impl_)
    return *impl_ == *other.impl_;
  else if(impl_)
    return isPointwise();
  else if(other.impl_)
    return other.isPointwise();
  else
    return true;
}
bool HorizontalExtent::operator!=(HorizontalExtent const& other) const { return !(*this == other); }
HorizontalExtent& HorizontalExtent::operator+=(HorizontalExtent const& other) {
  if(impl_ && other.impl_)
    *impl_ += *other.impl_;
  else if(other.impl_)
    *this = other;

  return *this;
}
void HorizontalExtent::merge(HorizontalExtent const& other) {
  if(impl_ && other.impl_)
    impl_->merge(*other.impl_);
  else if(impl_)
    impl_->addCenter();
  else if(other.impl_) {
    *this = other;
    impl_->addCenter();
  }
}
void HorizontalExtent::merge(ast::HorizontalOffset const& other) { merge(HorizontalExtent{other}); }
bool HorizontalExtent::isPointwise() const { return !impl_ || impl_->isPointwise(); }
void HorizontalExtent::limit(HorizontalExtent const& other) {
  if(impl_ && other.impl_)
    impl_->limit(*other.impl_);
  else if(!other.impl_)
    *this = other;
}

bool HorizontalExtent::hasType() const { return impl_ != nullptr; }

ast::GridType HorizontalExtent::getType() const {
  DAWN_ASSERT(hasType());
  if(dynamic_cast<CartesianExtent*>(impl_.get())) {
    return ast::GridType::Cartesian;
  } else {
    return ast::GridType::Unstructured;
  }
}

Extents::Extents() : Extents(HorizontalExtent{}, Extent{}) {}
Extents::Extents(HorizontalExtent const& hExtent, Extent const& vExtent)
    : verticalExtent_(vExtent), horizontalExtent_(hExtent) {}

Extents::Extents(ast::Offsets const& offset)
    : verticalExtent_(offset.verticalOffset()), horizontalExtent_(offset.horizontalOffset()) {}

Extents::Extents(ast::cartesian_, int extent1minus, int extent1plus, int extent2minus,
                 int extent2plus, int extent3minus, int extent3plus)
    : Extents(
          HorizontalExtent{ast::cartesian, extent1minus, extent1plus, extent2minus, extent2plus},
          Extent{extent3minus, extent3plus}) {}
Extents::Extents(ast::cartesian_) : horizontalExtent_(ast::cartesian) {}

Extents::Extents(ast::unstructured_, bool hasExtent, Extent const& vExtent)
    : Extents(HorizontalExtent{ast::unstructured, hasExtent}, Extent{vExtent}) {}
Extents::Extents(ast::unstructured_) : horizontalExtent_(ast::unstructured) {}

void Extents::merge(const Extents& other) {
  horizontalExtent_.merge(other.horizontalExtent_);
  verticalExtent_.merge(other.verticalExtent_);
}

void Extents::resetVerticalExtent() { verticalExtent_ = Extent(0, 0); }

void Extents::merge(const ast::Offsets& offset) {
  horizontalExtent_.merge(offset.horizontalOffset());
  verticalExtent_.merge(offset.verticalOffset());
}

Extents& Extents::operator+=(const Extents& other) {
  verticalExtent_ += other.verticalExtent_;
  horizontalExtent_ += other.horizontalExtent_;
  return *this;
}
Extents operator+(Extents lhs, const Extents& rhs) { return lhs += rhs; }
Extents merge(Extents lhs, Extents const& rhs) {
  lhs.merge(rhs);
  return lhs;
}
Extents limit(Extents lhs, Extents const& rhs) {
  lhs.limit(rhs);
  return lhs;
}

void Extents::addVerticalCenter() { verticalExtent_.merge(0); }

bool Extents::isHorizontalPointwise() const { return horizontalExtent_.isPointwise(); }

bool Extents::isVerticalPointwise() const { return verticalExtent_.isPointwise(); }

bool Extents::hasVerticalCenter() const {
  return verticalExtent_.minus() <= 0 && verticalExtent_.plus() >= 0;
}
void Extents::limit(Extents const& other) {
  horizontalExtent_.limit(other.horizontalExtent());
  verticalExtent_.limit(other.verticalExtent());
}

bool Extents::isPointwise() const {
  return horizontalExtent_.isPointwise() && verticalExtent_.isPointwise();
}

Extents::VerticalLoopOrderAccess
Extents::getVerticalLoopOrderAccesses(LoopOrderKind loopOrder) const {
  VerticalLoopOrderAccess access{false, false};

  if(isVerticalPointwise())
    return access;

  switch(loopOrder) {
  case LoopOrderKind::Parallel:
    // Any accesses in the vertical are against the loop-order
    return VerticalLoopOrderAccess{true, true};

  case LoopOrderKind::Forward: {
    // Accesses k+1 are against the loop order
    if(verticalExtent_.plus() > 0)
      access.CounterLoopOrder = true;
    if(verticalExtent_.minus() < 0)
      access.LoopOrder = true;
    break;
  }
  case LoopOrderKind::Backward:
    // Accesses k-1 are against the loop order
    if(verticalExtent_.minus() < 0)
      access.CounterLoopOrder = true;
    if(verticalExtent_.plus() > 0)
      access.LoopOrder = true;
    break;
  }

  return access;
}

std::optional<Extent> Extents::getVerticalLoopOrderExtent(LoopOrderKind loopOrder,
                                                          VerticalLoopOrderDir loopOrderPolicy,
                                                          bool includeCenter) const {
  if(loopOrder == LoopOrderKind::Parallel) {
    if(includeCenter && verticalExtent_.plus() >= 0 && verticalExtent_.minus() <= 0)
      return std::make_optional(Extent{0, 0});
    return std::optional<Extent>();
  }

  // retrieving the head (Plus) of the extent
  if((loopOrder == LoopOrderKind::Forward &&
      loopOrderPolicy == VerticalLoopOrderDir::CounterLoopOrder) ||
     (loopOrder == LoopOrderKind::Backward &&
      loopOrderPolicy == VerticalLoopOrderDir::InLoopOrder)) {
    if(verticalExtent_.plus() < (includeCenter ? 0 : 1))
      return std::optional<Extent>();

    // Accesses k+1 are against the loop order
    return std::make_optional(
        Extent{std::max((includeCenter ? 0 : 1), verticalExtent_.minus()), verticalExtent_.plus()});
  }
  // retrieving the tail (Minus) of the extent
  if((loopOrder == LoopOrderKind::Backward &&
      loopOrderPolicy == VerticalLoopOrderDir::CounterLoopOrder) ||
     (loopOrder == LoopOrderKind::Forward &&
      loopOrderPolicy == VerticalLoopOrderDir::InLoopOrder)) {
    if(verticalExtent_.minus() > (includeCenter ? 0 : -1))
      return std::optional<Extent>();

    // Accesses k-1 are against the loop order
    return std::make_optional(Extent{verticalExtent_.minus(),
                                     std::min((includeCenter ? 0 : -1), verticalExtent_.plus())});
  }
  dawn_unreachable("Non supported loop order");
}

bool Extents::operator==(const Extents& other) const {
  return verticalExtent_ == other.verticalExtent_ && horizontalExtent_ == other.horizontalExtent_;
}

bool Extents::operator!=(const Extents& other) const { return !(*this == other); }

std::string to_string(const Extents& extent) {
  auto const& vExtents = extent.verticalExtent();

  using namespace std::string_literals;
  return "["s +
         extent_dispatch(
             extent.horizontalExtent(),
             [&](CartesianExtent const& hExtents) {
               return "("s + std::to_string(hExtents.iMinus()) + "," +
                      std::to_string(hExtents.iPlus()) + "),(" + std::to_string(hExtents.jMinus()) +
                      "," + std::to_string(hExtents.jPlus()) + ")";
             },
             [&](UnstructuredExtent const& hExtents) {
               return hExtents.hasExtent() ? "<has_horizontal_extent>"s : "<no_horizontal_extent>"s;
             },
             [&]() { return "<no_horizontal_extent>"s; }) +
         ",(" + std::to_string(vExtents.minus()) + "," + std::to_string(vExtents.plus()) + ")]";
}

std::ostream& operator<<(std::ostream& os, const Extents& extents) {
  return os << to_string(extents);
}

} // namespace dawn::iir

namespace std {
size_t hash<dawn::iir::Extents>::operator()(const dawn::iir::Extents& extent) const {
  // TODO: Note this is only used from cartesian CodeGen right now. This will throw a bad_cast for
  // unstructured extents right now.
  auto const& hextent =
      dawn::iir::extent_cast<dawn::iir::CartesianExtent const&>(extent.horizontalExtent());
  auto const& vextent = extent.verticalExtent();

  size_t seed = 0;
  dawn::hash_combine(seed, hextent.iMinus(), hextent.iPlus(), hextent.jMinus(), hextent.jPlus(),
                     vextent.minus(), vextent.plus());
  return seed;
}
} // namespace std
