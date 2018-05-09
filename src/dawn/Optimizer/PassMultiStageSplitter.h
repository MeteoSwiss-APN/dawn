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

#ifndef DAWN_OPTIMIZER_PASSMULTISTAGESPLITTER_H
#define DAWN_OPTIMIZER_PASSMULTISTAGESPLITTER_H

#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief Pass for splitting multistages due vertical data dependencies
///
/// @see hasVerticalReadBeforeWriteConflict
/// @ingroup optimizer
class PassMultiStageSplitter : public Pass {
public:
  /// @brief Multistage splitting strategies
  enum MultiStageSplittingStrategy {
    SS_MaxCut, ///< Splitting the multistage into as many multistages as possible while maintaining
               /// code legality
    SS_Optimized ///< Optimized splitting of Multistages, only when needed
  };
  PassMultiStageSplitter(MultiStageSplittingStrategy strategy);

  /// @brief Pass implementation
  bool run(const std::shared_ptr<StencilInstantiation>& stencilInstantiation) override;

private:
  MultiStageSplittingStrategy strategy_;
};

} // namespace dawn

#endif
