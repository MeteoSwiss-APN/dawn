//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GSL_OPTIMIZER_RENAMING_H
#define GSL_OPTIMIZER_RENAMING_H

#include "gsl/Support/ArrayRef.h"
#include <memory>

namespace gsl {

class Expr;
class StatementAccessesPair;
class StencilFunctionInstantiation;
class StencilInstantiation;

/// @name Renaming routines
/// @ingroup optimizer
/// @{

/// @brief Rename all occurrences of `oldAccessID` to `newAccessID` in the `stmts` by updating the
/// AccessID maps
///
/// @param instantiation          Instantiation in which AccessID maps are updated
/// @param oldAccessID            Old AccessID of the field
/// @param newAccessID            New AccessID of the field
/// @param statementAccessesPair  AST statements to inspect
///
/// @ingroup optimizer
/// @{
void renameAccessIDInStmts(StencilInstantiation* instantiation, int oldAccessID, int newAccessID,
                           ArrayRef<std::shared_ptr<StatementAccessesPair>> statementAccessesPairs);
void renameAccessIDInStmts(StencilFunctionInstantiation* instantiation, int oldAccessID,
                           int newAccessID,
                           ArrayRef<std::shared_ptr<StatementAccessesPair>> statementAccessesPairs);
void renameAccessIDInExpr(StencilInstantiation* instantiation, int oldAccessID, int newAccessID,
                          std::shared_ptr<Expr>& expr);
/// @}

/// @brief Rename all occurrences of `oldAccessID` to `newAccessID` in the in the stencil or
/// stencil-function instantiation
///
/// For stencil-function instantiation the caller and callee accesses are renamed.
///
/// @param oldAccessID                  Old AccessID of the field
/// @param newAccessID                  New AccessID of the field
/// @param statementAccessesPairs       Accesses to update
///
/// @ingroup optimizer
void renameAccessIDInAccesses(
    StencilInstantiation* instantiation, int oldAccessID, int newAccessID,
    ArrayRef<std::shared_ptr<StatementAccessesPair>> statementAccessesPairs);
void renameAccessIDInAccesses(
    StencilFunctionInstantiation* instantiation, int oldAccessID, int newAccessID,
    ArrayRef<std::shared_ptr<StatementAccessesPair>> statementAccessesPairs);
/// @}

} // namespace gsl

#endif
