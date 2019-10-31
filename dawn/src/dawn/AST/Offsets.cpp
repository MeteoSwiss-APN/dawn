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

#ifndef DAWN_AST_OFFSETS_CPP
#define DAWN_AST_OFFSETS_CPP

#include "Offsets.h"

#include <iostream>

namespace dawn::ast {

std::string to_string(unstructured_, Offsets const& offset) {
  auto const& hoffset = offset_cast<UnstructuredOffset const&>(offset.horizontalOffset());
  auto const& voffset = offset.verticalOffset();

  using namespace std::string_literals;
  return (hoffset.hasOffset() ? "<has_horizontal_offset>"s : "<no_horizontal_offset>"s) + "," +
         std::to_string(voffset);
}

std::string to_string(cartesian_, Offsets const& offsets, std::string const& sep) {
  return to_string(cartesian, offsets, sep,
                   [](std::string const&, int offset) { return std::to_string(offset); });
}

std::string to_string(Offsets const& offset) {
  return offset_dispatch(offset.horizontalOffset(),
                         [&](CartesianOffset const&) { return to_string(cartesian, offset); },
                         [&](UnstructuredOffset const&) { return to_string(unstructured, offset); },
                         [&]() {
                           using namespace std::string_literals;
                           return "<no_horizontal_offset>, "s +
                                  std::to_string(offset.verticalOffset());
                         });
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
bool UnstructuredOffset::equalsImpl(HorizontalOffsetImpl const& other) const {
  auto const& uo_other = dynamic_cast<UnstructuredOffset const&>(other);
  return uo_other.hasOffset_ == hasOffset_;
}
void UnstructuredOffset::addImpl(HorizontalOffsetImpl const& other) {
  auto const& uo_other = dynamic_cast<UnstructuredOffset const&>(other);
  hasOffset_ = hasOffset_ || uo_other.hasOffset_;
}

Offsets operator+(Offsets o1, Offsets const& o2) { return o1 += o2; }
} // namespace dawn::ast

#endif
