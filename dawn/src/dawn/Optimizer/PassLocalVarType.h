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

#include "Pass.h"

namespace dawn {

/// @brief PassLocalVarType will compute, for each variable, its type
/// determining whether it is a scalar or it has an horizontal dimension and,
/// in the latter case, which dimension it is.
/// For unstructured grids this dimension (dense) can be either cells, vertices or edges.
/// For cartesian grids it's IJ.
/// @see LocalVariableType
/// * Input:  any IIR. Map from access id to LocalVariableData must be filled.
/// * Output: same as input but types of all local variables computed (LocalVariableData::type_ !=
///           std::nullopt)
/// @ingroup optimizer
///
/// This pass is necessary to generate legal IIR
class PassLocalVarType : public Pass {
public:
  PassLocalVarType(OptimizerContext& context) : Pass(context, "PassLocalVarType") {}

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn
