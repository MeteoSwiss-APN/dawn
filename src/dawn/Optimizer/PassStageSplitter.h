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

#ifndef DAWN_OPTIMIZER_PASSSTAGESPLITTER_H
#define DAWN_OPTIMIZER_PASSSTAGESPLITTER_H

#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief Pass for splitting stages due to horizontal non-pointwiese read-before-write data
/// dependencies
///
/// @see hasHorizontalReadBeforeWriteConflict
/// @ingroup optimizer
///
/// This pass is not necessary to create legal code and is hence not in the debug-group
class PassStageSplitter : public Pass {
public:
  PassStageSplitter();

  /// @brief Pass implementation
  bool run(const std::shared_ptr<StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn

#endif
