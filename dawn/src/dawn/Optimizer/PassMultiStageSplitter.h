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

/// @brief Pass for splitting multistages due vertical data dependencies
///
/// @see hasVerticalReadBeforeWriteConflict
/// @ingroup optimizer
class PassMultiStageSplitter : public Pass {
public:
  /// @brief Multistage splitting strategies
  enum class MultiStageSplittingStrategy {
    MaxCut,   ///< Splitting the multistage into as many multistages as possible while maintaining
              /// code legality
    Optimized ///< Optimized splitting of Multistages, only when needed
  };
  PassMultiStageSplitter(MultiStageSplittingStrategy strategy)
      : Pass("PassMultiStageSplitter"), strategy_(strategy) {}

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
           const Options& options = {}) override;

private:
  MultiStageSplittingStrategy strategy_;
};

} // namespace dawn
