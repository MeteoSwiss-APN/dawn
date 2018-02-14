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

#ifndef DAWN_OPTIMIZER_GREEDYCOLORING_H
#define DAWN_OPTIMIZER_GREEDYCOLORING_H

#include "dawn/Optimizer/DependencyGraphAccessesFwd.h"
#include <unordered_map>

namespace dawn {

/// @brief Greedily color the graph
class GreedyColoring {
  const DependencyGraphAccesses* graph_;
  std::unordered_map<int, int>& coloring_;

public:
  GreedyColoring(const DependencyGraphAccesses* graph, std::unordered_map<int, int>& coloring)
      : graph_(graph), coloring_(coloring) {}

  void compute();
};

} // namespace dawn

#endif
