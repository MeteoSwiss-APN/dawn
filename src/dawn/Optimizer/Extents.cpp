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
#include "dawn/Support/Unreachable.h"
#include "dawn/Support/StringUtil.h"
#include <iostream>

namespace dawn {

Extents::Extents(const Array3i& offset) {

  extents_ = std::array<Extent, 3>{};

  DAWN_ASSERT(extents_.size() == offset.size());

  for(std::size_t i = 0; i < extents_.size(); ++i) {
    extents_[i].Minus = offset[i];
    extents_[i].Plus = offset[i];
  }
}

Extents::Extents(int extent1Minus, int extent1Plus, int extent2Minus, int extent2Plus,
                 int extent3Minus, int extent3Plus) {
  extents_ =
      std::array<Extent, 3>({{Extent{extent1Minus, extent1Plus}, Extent{extent2Minus, extent2Plus},
                              Extent{extent3Minus, extent3Plus}}});
}

void Extents::addCenter(const unsigned int dim) {
  DAWN_ASSERT(dim < 3);

  extents_[dim].Minus = std::min(0, extents_[dim].Minus);
  extents_[dim].Plus = std::max(0, extents_[dim].Plus);
}

void Extents::merge(const Extents& other) {
  auto& extents = getExtents();

  DAWN_ASSERT(extents.size() == other.getExtents().size());
  for(std::size_t i = 0; i < extents.size(); ++i) {
    extents[i].merge(other.getExtents()[i]);
  }
}

void Extents::expand(const Extents& other) {
  auto& extents = getExtents();

  DAWN_ASSERT(extents.size() == other.getExtents().size());

  for(std::size_t i = 0; i < extents.size(); ++i)
    extents[i].expand(other.getExtents()[i]);
}

void Extents::merge(const Array3i& offset) {
  auto& extents = getExtents();
  DAWN_ASSERT(extents.size() == offset.size());

  for(std::size_t i = 0; i < extents.size(); ++i)
    extents[i].merge(offset[i] >= 0 ? Extent{0, offset[i]} : Extent{offset[i], 0});
}

void Extents::add(const Extents& other) {
  auto& extents = getExtents();
  DAWN_ASSERT(extents.size() == other.getExtents().size());

  for(std::size_t i = 0; i < extents.size(); ++i)
    extents[i].add(other.getExtents()[i]);
}

Extents Extents::add(const Extents& lhs, const Extents& rhs) {
  Extents sum = lhs;
  sum.add(rhs);
  return sum;
}

void Extents::add(const Array3i& offset) {
  auto& extents = getExtents();
  DAWN_ASSERT(extents.size() == offset.size());

  for(std::size_t i = 0; i < extents.size(); ++i)
    extents[i].add(offset[i]);
}

bool Extents::empty() {
  auto const& extents = getExtents();
  return extents.empty();
}

bool Extents::isPointwise() const {
  return isPointwiseInDim(0) && isPointwiseInDim(1) && isPointwiseInDim(2);
}

bool Extents::isPointwiseInDim(int dim) const {
  auto const& extents = getExtents();
  return (extents[dim].Minus == 0 && extents[dim].Plus == 0);
}

bool Extents::isHorizontalPointwise() const { return isPointwiseInDim(0) && isPointwiseInDim(1); }
bool Extents::isVerticalPointwise() const { return isPointwiseInDim(2); }

Extents::VerticalLoopOrderAccess
Extents::getVerticalLoopOrderAccesses(LoopOrderKind loopOrder) const {
  auto const& extents = getExtents();
  VerticalLoopOrderAccess access{false, false};

  if(isVerticalPointwise())
    return access;

  const Extent& verticalExtent = extents[2];

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
                                                            VerticalLoopOrderDir loopOrderDir,
                                                            bool includeCenter) const {
  const auto& extents = getExtents();
  const Extent& verticalExtent = extents[2];

  if(loopOrder == LoopOrderKind::LK_Parallel) {
    if(includeCenter && verticalExtent.Plus >= 0 && verticalExtent.Minus <= 0)
      return boost::make_optional(Extent{0, 0});
    return boost::optional<Extent>();
  }
  if((loopOrder == LoopOrderKind::LK_Forward &&
      loopOrderDir == VerticalLoopOrderDir::VL_CounterLoopOrder) ||
     (loopOrder == LoopOrderKind::LK_Backward &&
      loopOrderDir == VerticalLoopOrderDir::VL_InLoopOrder)) {
    if(verticalExtent.Plus < (includeCenter ? 0 : 1))
      return boost::optional<Extent>();

    // Accesses k+1 are against the loop order
    return boost::make_optional(
        Extent{std::max((includeCenter ? 0 : 1), verticalExtent.Minus), verticalExtent.Plus});
  }
  if((loopOrder == LoopOrderKind::LK_Backward &&
      loopOrderDir == VerticalLoopOrderDir::VL_CounterLoopOrder) ||
     (loopOrder == LoopOrderKind::LK_Forward &&
      loopOrderDir == VerticalLoopOrderDir::VL_InLoopOrder)) {
    if(verticalExtent.Minus > (includeCenter ? 0 : -1))
      return boost::optional<Extent>();

    // Accesses k-1 are against the loop order
    return boost::make_optional(
        Extent{verticalExtent.Minus, std::min((includeCenter ? 0 : -1), verticalExtent.Plus)});
  }
  dawn_unreachable("Non supported loop order");
}

bool Extents::operator==(const Extents& other) const {
  auto const& extents = getExtents();
  return (extents[0] == other[0] && extents[1] == other[1] && extents[2] == other[2]);
}

bool Extents::operator!=(const Extents& other) const { return !(*this == other); }

std::ostream& operator<<(std::ostream& os, const Extents& extent) {
  return (os << RangeToString()(extent.getExtents(), [](const Extent& e) -> std::string {
            return "(" + std::to_string(e.Minus) + ", " + std::to_string(e.Plus) + ")";
          }));
}

} // namespace dawn
