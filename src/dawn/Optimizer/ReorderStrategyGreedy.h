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

#ifndef DAWN_OPTIMIZER_REORDERSTRATEGYGREEDY_H
#define DAWN_OPTIMIZER_REORDERSTRATEGYGREEDY_H

#include "dawn/Optimizer/ReorderStrategy.h"

namespace dawn {

/// @brief Reordering strategy which tries to move each stage upwards as far as possible under the
/// sole constraint that the extent of any field does not exeed the maximum halo points
/// @ingroup optimizer
class ReoderStrategyGreedy : public ReorderStrategy {
public:
  /// @brief Apply the reordering strategy and return stencil
  virtual std::shared_ptr<Stencil> reorder(const std::shared_ptr<Stencil>& stencilPtr) override;
};

} // namespace dawn

#endif
