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

#include "dawn/Optimizer/Options.h"

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

  enum class Kind {
    Unknown,
    None,         ///< Don't perform any reordering
    Greedy,       ///< Greedy fusing of the stages until max-halo boundary is reached
    Partitioning, ///< Use S-cut graph partitioning
    Permutations  ///< Use Mating and Mutations for genetic Algorithms (this required an
                  /// existing, correct IIR
  };

  /// @brief Reorder the stages of the `stencilPtr` according to the implemented strategy
  /// @returns New stencil with the reordered stages
  virtual std::unique_ptr<iir::Stencil> reorder(iir::StencilInstantiation* instantiation,
                                                const std::unique_ptr<iir::Stencil>& stencilPtr,
                                                const Options& options) = 0;
};

} // namespace dawn
