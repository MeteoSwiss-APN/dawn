//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gsl/Optimizer/Extents.h"
#include "gsl/Support/StringUtil.h"
#include <iostream>

namespace gsl {

Extents::Extents() {}

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

void Extents::merge(const Extents& other) {
  for(std::size_t i = 0; i < extents_.size(); ++i)
    extents_[i].merge(other.extents_[i]);
}

void Extents::merge(const Array3i& offset) {
  for(std::size_t i = 0; i < extents_.size(); ++i)
    extents_[i].merge(offset[i] >= 0 ? Extent{0, offset[i]} : Extent{offset[i], 0});
}

void Extents::add(const Extents& other) {
  for(std::size_t i = 0; i < extents_.size(); ++i)
    extents_[i].add(other.extents_[i]);
}

Extents Extents::add(const Extents& lhs, const Extents& rhs) {
  Extents sum;
  for(std::size_t i = 0; i < sum.extents_.size(); ++i)
    sum.extents_[i] = Extent::add(lhs.extents_[i], rhs.extents_[i]);
  return sum;
}

void Extents::add(const Array3i& offset) {
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

bool Extents::operator==(const Extents& other) const {
  return (extents_[0] == other[0] && extents_[1] == other[1] && extents_[2] == other[2]);
}

bool Extents::operator!=(const Extents& other) const { return !(*this == other); }

std::ostream& operator<<(std::ostream& os, const Extents& extent) {
  return (os << RangeToString()(extent.getExtents(), [](const Extent& e) -> std::string {
            return "(" + std::to_string(e.Minus) + ", " + std::to_string(e.Plus) + ")";
          }));
}

} // namespace gsl
