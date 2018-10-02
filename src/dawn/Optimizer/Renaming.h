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

#ifndef DAWN_OPTIMIZER_RENAMING_H
#define DAWN_OPTIMIZER_RENAMING_H

#include "dawn/Support/ArrayRef.h"
#include <memory>

namespace dawn {

class Expr;
namespace iir {
class StatementAccessesPair;
class StencilFunctionInstantiation;
class IIR;
}

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
void renameAccessIDInStmts(
    iir::IIR *iir, int oldAccessID, int newAccessID,
    ArrayRef<std::unique_ptr<iir::StatementAccessesPair>> statementAccessesPairs);
void renameAccessIDInExpr(iir::IIR *iir, int oldAccessID,
                          int newAccessID, std::shared_ptr<Expr>& expr);
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
//void renameAccessIDInAccesses(
//    iir::StencilInstantiation* instantiation, int oldAccessID, int newAccessID,
//    ArrayRef<std::unique_ptr<iir::StatementAccessesPair>> statementAccessesPairs);
void renameAccessIDInAccesses(int oldAccessID, int newAccessID,
    ArrayRef<std::unique_ptr<iir::StatementAccessesPair>> statementAccessesPairs);
/// @}

} // namespace dawn

#endif
