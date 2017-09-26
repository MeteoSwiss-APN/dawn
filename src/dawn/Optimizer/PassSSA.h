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

#ifndef DAWN_OPTIMIZER_PASSSSA_H
#define DAWN_OPTIMIZER_PASSSSA_H

#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief Converts each DAG of a stencil into SSA form (Static Single Assignment)
///
/// @see https://en.wikipedia.org/wiki/Static_single_assignment_form
/// @ingroup optimizer
class PassSSA : public Pass {
public:
  PassSSA();

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;
};

} // namespace dawn

#endif
