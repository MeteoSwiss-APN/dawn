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

#pragma once

#include <vector>

namespace dawn {
namespace ast {

enum class LocationType { Cells, Edges, Vertices };
using NeighborChain = std::vector<ast::LocationType>;

} // namespace ast
} // namespace dawn

namespace std {
template <>
struct hash<std::vector<dawn::ast::LocationType>> {
  size_t operator()(const std::vector<dawn::ast::LocationType>& vec) const {
    std::size_t seed = vec.size();
    for(auto& i : vec) {
      seed ^= (int)i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};
} // namespace std