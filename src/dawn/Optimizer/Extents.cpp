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

#include "dawn/Optimizer/Extents.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/StringUtil.h"
#include <iostream>

namespace dawn {

Extents::Extents() : extents_{} {}

Extents::Extents(const Array3i& offset) { merge(offset); }

Extents::Extents(int extent1Minus, int extent1Plus, int extent2Minus, int extent2Plus,
                 int extent3Minus, int extent3Plus) {
  extents_[0].Minus = extent1Minus;
  extents_[0].Plus = extent1Plus;
  extents_[1].Minus = extent2Minus;
  extents_[1].Plus = extent2Plus;
  extents_[2].Minus = extent3Minus;
  extents_[2].Plus = extent3Plus;
}

void Extents::addCenter(const unsigned int dim) {
  DAWN_ASSERT(dim < 3);

  extents_[dim].Minus = std::min(0, extents_[dim].Minus);
  extents_[dim].Plus = std::max(0, extents_[dim].Plus);
}

void Extents::merge(const Extents& other) {
  DAWN_ASSERT(extents_.size() == other.getExtents().size());
  for(std::size_t i = 0; i < extents_.size(); ++i) {
    extents_[i].merge(other.getExtents()[i]);
  }
}

void Extents::expand(const Extents& other) {
  DAWN_ASSERT(extents_.size() == other.getExtents().size());

  for(std::size_t i = 0; i < extents_.size(); ++i)
    extents_[i].expand(other.getExtents()[i]);
}

void Extents::merge(const Array3i& offset) {
  DAWN_ASSERT(extents_.size() == offset.size());

  for(std::size_t i = 0; i < extents_.size(); ++i)
    extents_[i].merge(offset[i] >= 0 ? Extent{0, offset[i]} : Extent{offset[i], 0});
}

void Extents::add(const Extents& other) {
  DAWN_ASSERT(extents_.size() == other.getExtents().size());

  for(std::size_t i = 0; i < extents_.size(); ++i)
    extents_[i].add(other.getExtents()[i]);
}

Extents Extents::add(const Extents& lhs, const Extents& rhs) {
  Extents sum;
  for(std::size_t i = 0; i < sum.extents_.size(); ++i)
    sum.extents_[i] = Extent::add(lhs.extents_[i], rhs.extents_[i]);
  return sum;
}

void Extents::add(const Array3i& offset) {
  DAWN_ASSERT(extents_.size() == offset.size());

  for(std::size_t i = 0; i < extents_.size(); ++i)
    extents_[i].add(offset[i]);
}

bool Extents::empty() { return extents_.empty(); }

bool Extents::isPointwise() const {
  return isPointwiseInDim(0) && isPointwiseInDim(1) && isPointwiseInDim(2);
}

bool Extents::isPointwiseInDim(int dim) const {
  return (extents_[dim].Minus == 0 && extents_[dim].Plus == 0);
}

bool Extents::isHorizontalPointwise() const { return isPointwiseInDim(0) && isPointwiseInDim(1); }
bool Extents::isVerticalPointwise() const { return isPointwiseInDim(2); }

Extents::VerticalLoopOrderAccess
Extents::getVerticalLoopOrderAccesses(LoopOrderKind loopOrder) const {
  VerticalLoopOrderAccess access{false, false};

  if(isVerticalPointwise())
    return access;

  const Extent& verticalExtent = extents_[2];

  switch(loopOrder) {
  case LoopOrderKind::LK_Parallel:
    // Any accesses in the vertical are against the loop-order
    return VerticalLoopOrderAccess{true, true};

  case LoopOrderKind::LK_Forward: {
    // Accesses k+1 are against the loop order
    if(verticalExtent.Plus > 0)
      access.CounterLoopOrder = true;
    if(verticalExtent.Minus < 0)
      access.LoopOrder = true;
    break;
  }
  case LoopOrderKind::LK_Backward:
    // Accesses k-1 are against the loop order
    if(verticalExtent.Minus < 0)
      access.CounterLoopOrder = true;
    if(verticalExtent.Plus > 0)
      access.LoopOrder = true;
    break;
  }

  return access;
}

boost::optional<Extent> Extents::getVerticalLoopOrderExtent(LoopOrderKind loopOrder,
                                                            VerticalLoopOrderDir loopOrderPolicy,
                                                            bool includeCenter) const {
  const Extent& verticalExtent = extents_[2];

  if(loopOrder == LoopOrderKind::LK_Parallel) {
    if(includeCenter && verticalExtent.Plus >= 0 && verticalExtent.Minus <= 0)
      return boost::make_optional(Extent{0, 0});
    return boost::optional<Extent>();
  }

  // retrieving the head (Plus) of the extent
  if((loopOrder == LoopOrderKind::LK_Forward &&
      loopOrderPolicy == VerticalLoopOrderDir::VL_CounterLoopOrder) ||
     (loopOrder == LoopOrderKind::LK_Backward &&
      loopOrderPolicy == VerticalLoopOrderDir::VL_InLoopOrder)) {
    if(verticalExtent.Plus < (includeCenter ? 0 : 1))
      return boost::optional<Extent>();

    // Accesses k+1 are against the loop order
    return boost::make_optional(
        Extent{std::max((includeCenter ? 0 : 1), verticalExtent.Minus), verticalExtent.Plus});
  }
  // retrieving the tail (Minus) of the extent
  if((loopOrder == LoopOrderKind::LK_Backward &&
      loopOrderPolicy == VerticalLoopOrderDir::VL_CounterLoopOrder) ||
     (loopOrder == LoopOrderKind::LK_Forward &&
      loopOrderPolicy == VerticalLoopOrderDir::VL_InLoopOrder)) {
    if(verticalExtent.Minus > (includeCenter ? 0 : -1))
      return boost::optional<Extent>();

    // Accesses k-1 are against the loop order
    return boost::make_optional(
        Extent{verticalExtent.Minus, std::min((includeCenter ? 0 : -1), verticalExtent.Plus)});
  }
  dawn_unreachable("Non supported loop order");
}

bool Extents::operator==(const Extents& other) const {
  return (extents_[0] == other[0] && extents_[1] == other[1] && extents_[2] == other[2]);
}

bool Extents::operator!=(const Extents& other) const { return !(*this == other); }

std::ostream& operator<<(std::ostream& os, const Extents& extent) {
  return (os << RangeToString()(extent.getExtents(), [](const Extent& e) -> std::string {
            return "(" + std::to_string(e.Minus) + ", " + std::to_string(e.Plus) + ")";
          }));
}

} // namespace dawn
