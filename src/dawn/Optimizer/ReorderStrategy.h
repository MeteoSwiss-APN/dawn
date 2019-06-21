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

#ifndef DAWN_OPTIMIZER_REORDERSTRATEGY_H
#define DAWN_OPTIMIZER_REORDERSTRATEGY_H

#include <memory>

namespace dawn {

namespace iir {
class Stencil;
class StencilInstantiation;
class DependencyGraphStage;
} // namespace iir

/// @brief Abstract class for various reodering strategies
/// @ingroup optimizer
class ReorderStrategy {
public:
  virtual ~ReorderStrategy() {}

  enum ReorderStrategyKind {
    RK_Unknown,
    RK_None,         ///< Don't perform any reordering
    RK_Greedy,       ///< Greedy fusing of the stages until max-halo boundary is reached
    RK_Partitioning, ///< Use S-cut graph partitioning
    RK_Permutations  ///< Use Mating and Mutations for genetic Algorithms (this required an
                     /// existing, correct IIR
  };

  /// @brief Reorder the stages of the `stencilPtr` according to the implemented strategy
  /// @returns New stencil with the reordered stages
  virtual std::unique_ptr<iir::Stencil>
  reorder(iir::StencilInstantiation* instantiation,
          const std::unique_ptr<iir::Stencil>& stencilPtr) = 0;
};

} // namespace dawn

#endif
