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

#ifndef GSL_OPTIMIZER_REORDERSTRATEGY_H
#define GSL_OPTIMIZER_REORDERSTRATEGY_H

#include <memory>

namespace gsl {

class Stencil;
class DependencyGraphStage;

/// @brief Abstract class for various reodering strategies
/// @ingroup optimizer
class ReorderStrategy {
public:
  enum ReorderStrategyKind {
    RK_Unknown,
    RK_None,        ///< Don't perform any reordering
    RK_Greedy,      ///< Greedy fusing of the stages until max-halo boundary is reached
    RK_Partitioning ///< Use S-cut graph partitioning
  };

  /// @brief Reorder the stages of the `stencilPtr` according to the implemented strategy
  /// @returns New stencil with the reordered stages
  virtual std::shared_ptr<Stencil> reorder(const std::shared_ptr<Stencil>& stencilPtr) = 0;
};

} // namespace gsl

#endif
