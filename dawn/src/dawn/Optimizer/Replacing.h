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
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/Support/ArrayRef.h"
#include <memory>

namespace dawn {

struct Statement;

namespace iir {
class Stencil;
class StencilInstantiation;
class StencilMetaInformation;
} // namespace iir

/// @name Replacing routines
/// @ingroup optimizer
/// @{

/// @brief Replace all field accesses with variable accesses in the given `stmts`
///
/// This will also modify the underlying AccessID maps of the StencilInstantiation.
void replaceFieldWithVarAccessInStmts(iir::Stencil* stencil, int AccessID,
                                      const std::string& varname,
                                      ArrayRef<std::shared_ptr<iir::Stmt>> stmts);

/// @brief Replace all variable accesses with field accesses in the given `stmts`
///
/// This will also modify the underlying AccessID maps of the StencilInstantiation.
void replaceVarWithFieldAccessInStmts(iir::Stencil* stencil, int AccessID,
                                      const std::string& fieldname,
                                      ArrayRef<std::shared_ptr<iir::Stmt>> stmts);

/// @brief Replace all stencil calls to `oldStencilID` with a series of stencil calls to
/// `newStencilIDs` in the stencil description AST of `instantiation`
void replaceStencilCalls(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                         int oldStencilID, const std::vector<int>& newStencilIDs);

/// @}

} // namespace dawn
