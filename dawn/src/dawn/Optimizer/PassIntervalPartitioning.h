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
#include "dawn/Support/Assert.h"
#include <set>
#include <unordered_map>

namespace dawn {

namespace iir {
class Stencil;
class DoMethod;
} // namespace iir

/// @brief PassTemporaryToStencilFunction pass will identify temporaries of a stencil and replace
/// their pre-computations
/// by a stencil function. Each reference to the temporary is later replaced by the stencil function
/// call.
/// * Input: well formed SIR and IIR with the list of mss/stages, temporaries used
/// * Output: modified SIR, new stencil functions are inserted and calls. Temporary fields are
/// removed. New stencil functions instantiations are inserted into the IIR. Statements' accesses
/// are recomputed.
/// @ingroup optimizer
///
/// This pass is not necessary to create legal code and is hence not in the debug-group
class PassIntervalPartitioning : public Pass {
public:
  PassIntervalPartitioning(OptimizerContext& context) : Pass(context, "PassIntervalPartitioning") {}

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn
