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

#ifndef DAWN_OPTIMIZER_REPLACING_H
#define DAWN_OPTIMIZER_REPLACING_H

#include "dawn/SIR/ASTVisitor.h"
#include "dawn/Support/ArrayRef.h"
#include <memory>

namespace dawn {

class Stencil;
struct Statement;
class StatementAccessesPair;
class StencilInstantiation;

/// @brief The GetStencilCalls class reads the StencilDescAST and finds all the stencils with given
/// ID.
class GetStencilCalls : public ASTVisitorForwarding {
  StencilInstantiation* instantiation_;
  int StencilID_;

  std::vector<std::shared_ptr<StencilCallDeclStmt>> stencilCallsToReplace_;

public:
  GetStencilCalls(StencilInstantiation* instantiation, int StencilID);

  void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override;

  std::vector<std::shared_ptr<StencilCallDeclStmt>>& getStencilCallsToReplace();

  void reset();
};

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

} // namespace dawn

#endif
