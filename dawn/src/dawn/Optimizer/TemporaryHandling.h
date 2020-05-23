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
//===------------------------------------------------------------------------------------------===/
#pragma once
#include "dawn/IIR/StencilInstantiation.h"

namespace dawn {

/// @brief Promote the local variable, given by `AccessID`, to a temporary field
///
/// This will take care of registering the new field (and removing the variable) as well as
/// replacing the variable accesses with point-wise field accesses.
void promoteLocalVariableToTemporaryField(iir::StencilInstantiation* instantiation,
                                          iir::Stencil* stencil, int accessID,
                                          const iir::Stencil::Lifetime& lifetime,
                                          iir::TemporaryScope temporaryScope);

/// @brief Promote the temporary field, given by `AccessID`, to a real storage which needs to be
/// allocated by the stencil
void promoteTemporaryFieldToAllocatedField(iir::StencilInstantiation* instantiation, int AccessID);

// @brief Demote the temporary field, given by `AccessID`, to a local variable
///
/// This will take care of registering the new variable (and removing the field) as well as
/// replacing the field accesses with varible accesses.
///
/// This implicitcly assumes the first access (i.e `lifetime.Begin`) to the field is an
/// `ExprStmt` and the field is accessed as the LHS of an `AssignmentExpr`.
void demoteTemporaryFieldToLocalVariable(iir::StencilInstantiation* instantiation,
                                         iir::Stencil* stencil, int AccessID,
                                         const iir::Stencil::Lifetime& lifetime);
} // namespace dawn
