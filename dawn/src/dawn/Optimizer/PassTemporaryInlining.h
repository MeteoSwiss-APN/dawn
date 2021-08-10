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

#include "dawn/Optimizer/Pass.h"

namespace dawn {
/// @brief Try to inline computation of stencil temporaries
///
/// this is an experimental pass and very limited in functionality. All that is inlined currently
/// are temporaries holding the result of a reduceOverNeighborExpr with a single use, e.g.
///
/// tempF = sum_over(Edge > Cell, inF)
/// outF = sum_over(Vertex > Edge, tempF)
///
/// is inlined to
///
/// outF = sum_over(Vertex > Edge, sum_over(Edge > Cell, inF))
///
/// @ingroup optimizer
///
/// This pass is not necessary to create legal code and is hence not in the debug-group
class PassTemporaryInlining : public Pass {
public:
  PassTemporaryInlining() : Pass("PassTemporaryInlining") {}

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
           const Options& options = {}) override;
};

} // namespace dawn
