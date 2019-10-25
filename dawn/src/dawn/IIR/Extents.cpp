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
#include "dawn/Support/Assert.h"
#include "dawn/Support/StringUtil.h"
#include "dawn/Support/Unreachable.h"
#include <algorithm>
#include <iostream>

namespace dawn {
namespace iir {

Extent operator+(Extent lhs, Extent const& rhs) { return lhs += rhs; }

Extents::Extents() : Extents(HorizontalExtent{}, Extent{}) {}
Extents::Extents(HorizontalExtent const& hExtent, Extent const& vExtent)
    : verticalExtent_(vExtent), horizontalExtent_(hExtent) {}

Extents::Extents(const ast::Offsets& offset)
    : verticalExtent_(offset.verticalOffset(), offset.verticalOffset()),
      horizontalExtent_(offset.horizontalOffset()) {}

Extents::Extents(ast::cartesian_, int extent1minus, int extent1plus, int extent2minus,
                 int extent2plus, int extent3minus, int extent3plus)
    : verticalExtent_(extent3minus, extent3plus),
      horizontalExtent_(ast::cartesian, extent1minus, extent1plus, extent2minus, extent2plus) {}

Extents::Extents(ast::cartesian_) : horizontalExtent_(ast::cartesian) {}

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

void Extents::addVerticalCenter() { verticalExtent_.merge(0); }

bool Extents::isHorizontalPointwise() const { return horizontalExtent_.isPointwise(); }

bool Extents::isVerticalPointwise() const { return verticalExtent_.isPointwise(); }

bool Extents::hasVerticalCenter() const {
  return verticalExtent_.minus() <= 0 && verticalExtent_.plus() >= 0;
}
Extents Extents::limit(Extents const& other) const {
  return Extents{horizontalExtent_.limit(other.horizontalExtent()),
                 verticalExtent_.limit(other.verticalExtent())};
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
  case LoopOrderKind::LK_Parallel:
    // Any accesses in the vertical are against the loop-order
    return VerticalLoopOrderAccess{true, true};

  case LoopOrderKind::LK_Forward: {
    // Accesses k+1 are against the loop order
    if(verticalExtent_.plus() > 0)
      access.CounterLoopOrder = true;
    if(verticalExtent_.minus() < 0)
      access.LoopOrder = true;
    break;
  }
  case LoopOrderKind::LK_Backward:
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
  if(loopOrder == LoopOrderKind::LK_Parallel) {
    if(includeCenter && verticalExtent_.plus() >= 0 && verticalExtent_.minus() <= 0)
      return std::make_optional(Extent{0, 0});
    return std::optional<Extent>();
  }

  // retrieving the head (Plus) of the extent
  if((loopOrder == LoopOrderKind::LK_Forward &&
      loopOrderPolicy == VerticalLoopOrderDir::VL_CounterLoopOrder) ||
     (loopOrder == LoopOrderKind::LK_Backward &&
      loopOrderPolicy == VerticalLoopOrderDir::VL_InLoopOrder)) {
    if(verticalExtent_.plus() < (includeCenter ? 0 : 1))
      return std::optional<Extent>();

    // Accesses k+1 are against the loop order
    return std::make_optional(
        Extent{std::max((includeCenter ? 0 : 1), verticalExtent_.minus()), verticalExtent_.plus()});
  }
  // retrieving the tail (Minus) of the extent
  if((loopOrder == LoopOrderKind::LK_Backward &&
      loopOrderPolicy == VerticalLoopOrderDir::VL_CounterLoopOrder) ||
     (loopOrder == LoopOrderKind::LK_Forward &&
      loopOrderPolicy == VerticalLoopOrderDir::VL_InLoopOrder)) {
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
  auto const& hExtents = extent_cast<CartesianExtent const&>(extent.horizontalExtent());
  auto const& vExtents = extent.verticalExtent();

  return "[(" + std::to_string(hExtents.iMinus()) + ", " + std::to_string(hExtents.iPlus()) +
         "), (" + std::to_string(hExtents.jMinus()) + ", " + std::to_string(hExtents.jPlus()) +
         "), (" + std::to_string(vExtents.minus()) + ", " + std::to_string(vExtents.plus()) + ")]";
}

std::ostream& operator<<(std::ostream& os, const Extents& extents) {
  return os << to_string(extents);
}

} // namespace iir
} // namespace dawn
