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

#include "dawn/IIR/ASTFwd.h"
#include "dawn/IIR/DoMethod.h"
#include "dawn/Support/ArrayRef.h"
#include <memory>

namespace dawn {

namespace iir {
class StencilFunctionInstantiation;
class StencilInstantiation;
class StencilMetaInformation;
class MultiStage;
class Stencil;
} // namespace iir

/// @name Renaming routines
/// @ingroup optimizer
/// @{

/// @brief Rename all occurrences of `oldAccessID` to `newAccessID` in the `stmts` by updating the
/// AccessID maps
///
/// @param instantiation          Instantiation in which AccessID maps are updated
/// @param oldAccessID            Old AccessID of the field
/// @param newAccessID            New AccessID of the field
/// @param stmt  AST statements to inspect
///
/// @ingroup optimizer
/// @{
void renameAccessIDInStmts(iir::StencilMetaInformation* instantiation, int oldAccessID,
                           int newAccessID, ArrayRef<std::shared_ptr<iir::Stmt>> stmts);
void renameAccessIDInStmts(iir::StencilFunctionInstantiation* instantiation, int oldAccessID,
                           int newAccessID, ArrayRef<std::shared_ptr<iir::Stmt>> stmts);
void renameAccessIDInExpr(iir::StencilInstantiation* instantiation, int oldAccessID,
                          int newAccessID, std::shared_ptr<iir::Expr>& expr);
/// @}

/// @brief Rename all occurrences of `oldAccessID` to `newAccessID` in the in the stencil or
/// stencil-function instantiation
///
/// For stencil-function instantiation the caller and callee accesses are renamed.
///
/// @param oldAccessID                  Old AccessID of the field
/// @param newAccessID                  New AccessID of the field
/// @param stmts       Accesses to update
///
/// @ingroup optimizer
void renameAccessIDInAccesses(const iir::StencilMetaInformation* metadata, int oldAccessID,
                              int newAccessID, ArrayRef<std::shared_ptr<iir::Stmt>> stmts);
void renameAccessIDInAccesses(iir::StencilFunctionInstantiation* instantiation, int oldAccessID,
                              int newAccessID, ArrayRef<std::shared_ptr<iir::Stmt>> stmts);
/// @}

void renameAccessIDInMultiStage(iir::MultiStage* multiStage, int oldAccessID, int newAccessID);

void renameAccessIDInStencil(iir::Stencil* stencil, int oldAccessID, int newAccessID);

/// @brief Rename all occurences of the caller AccessID from `oldAccessID` to `newAccessID`
void renameCallerAccessIDInStencilFunction(iir::StencilFunctionInstantiation* function,
                                           int oldAccessID, int newAccessID);
} // namespace dawn
