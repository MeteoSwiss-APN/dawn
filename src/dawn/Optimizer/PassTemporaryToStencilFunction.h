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

#ifndef DAWN_OPTIMIZER_PASSTEMPORARYTOSTENCILFUNCTION_H
#define DAWN_OPTIMIZER_PASSTEMPORARYTOSTENCILFUNCTION_H

#include "dawn/Optimizer/Pass.h"

namespace dawn {

class Stencil;
class DoMethod;

/// @brief PassTemporaryToStencilFunction pass will identify temporaries of a stencil and replace
/// their pre-computations
/// by a stencil function. Each reference to the temporary is later replaced by the stencil function
/// call.
/// * Input: well formed SIR and IIR with the list of mss/stages, temporaries used and
/// <statement,accesses> pairs
/// * Output: modified SIR, new stencil functions are inserted and calls. Temporary fields are
/// removed. New stencil functions instantiations are inserted into the IIR. <statement,accesses>
/// pairs are recomputed
/// @ingroup optimizer
///
/// This pass is not necessary to create legal code and is hence not in the debug-group
class PassTemporaryToStencilFunction : public Pass {

public:
  PassTemporaryToStencilFunction();

  /// @brief Pass implementation
  bool run(const std::shared_ptr<StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn

#endif
