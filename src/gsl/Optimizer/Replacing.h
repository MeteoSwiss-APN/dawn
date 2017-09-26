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

#ifndef GSL_OPTIMIZER_REPLACING_H
#define GSL_OPTIMIZER_REPLACING_H

#include "gsl/Support/ArrayRef.h"
#include <memory>

namespace gsl {

class Stencil;
struct Statement;
class StatementAccessesPair;
class StencilInstantiation;

/// @name Replacing routines
/// @ingroup optimizer
/// @{

/// @brief Replace all field accesses with variable accesses in the given `stmts`
///
/// This will also modify the underlying AccessID maps of the StencilInstantiation.
void replaceFieldWithVarAccessInStmts(
    Stencil* stencil, int AccessID, const std::string& varname,
    ArrayRef<std::shared_ptr<StatementAccessesPair>> statementAccessesPairs);

/// @brief Replace all variable accesses with field accesses in the given `stmts`
///
/// This will also modify the underlying AccessID maps of the StencilInstantiation.
void replaceVarWithFieldAccessInStmts(
    Stencil* stencil, int AccessID, const std::string& fieldname,
    ArrayRef<std::shared_ptr<StatementAccessesPair>> statementAccessesPairs);

/// @brief Replace all stencil calls to `oldStencilID` with a series of stencil calls to
/// `newStencilIDs` in the stencil description AST of `instantiation`
void replaceStencilCalls(StencilInstantiation* instantiation, int oldStencilID,
                        const std::vector<int>& newStencilIDs);

/// @}

} // namespace gsl

#endif
