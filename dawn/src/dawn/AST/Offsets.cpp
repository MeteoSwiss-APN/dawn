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

std::ostream& operator<<(std::ostream& os, Offsets const& offset) { return os << toString(offset); }

std::string toString(Offsets const& offsets, std::string const& sep) {
  return toString(offsets, sep,
                  [](std::string const&, int offset) { return std::to_string(offset); });
}

bool StructuredOffset::equalsImpl(HorizontalOffsetImpl const& other) const {
  auto const& so_other = dynamic_cast<StructuredOffset const&>(other);
  return so_other.horizontalOffset_ == horizontalOffset_;
}

StructuredOffset& StructuredOffset::addImpl(HorizontalOffsetImpl const& other) {
  auto const& so_other = dynamic_cast<StructuredOffset const&>(other);
  horizontalOffset_[0] += so_other.horizontalOffset_[0];
  horizontalOffset_[1] += so_other.horizontalOffset_[1];
  return *this;
}

} // namespace dawn::ast

#endif
