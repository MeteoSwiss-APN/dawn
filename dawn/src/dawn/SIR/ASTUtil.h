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

#include "dawn/AST/ASTUtil.h"
#include "dawn/SIR/ASTFwd.h"

namespace dawn {
namespace sir {
/// @brief Replace `oldExpr` with `newExpr` in `stmt`
///
/// Note that `oldExpr` is identfied by its `std::shared_ptr` and not its content. To replace
/// fields or variables in expr see `Optimizer/Replacing.cpp`.
///
/// @param stmt     Statement to analyze
/// @param oldExpr  Expression to replace
/// @param newExpr  Expression to use as a replacement
///
/// @ingroup sir
extern void replaceOldExprWithNewExprInStmt(const std::shared_ptr<Stmt>& stmt,
                                            const std::shared_ptr<Expr>& oldExpr,
                                            const std::shared_ptr<Expr>& newExpr);

/// @brief Replace `oldStmt` with `newStmt` in `stmt`
///
/// Note that `oldStmt` is identfied by its `std::shared_ptr` and not its content.
///
/// @param stmt     Statement to analyze
/// @param oldStmt  Statement to replace
/// @param newStmt  Statement to use as a replacement
///
/// @ingroup sir
extern void replaceOldStmtWithNewStmtInStmt(const std::shared_ptr<Stmt>& stmt,
                                            const std::shared_ptr<Stmt>& oldStmt,
                                            const std::shared_ptr<Stmt>& newStmt);

/// @brief Try to evaluate the expression `expr`
///
/// Expressions can only be evaluated if they consist of unary, binary or ternary operators on
/// literals. Variable accesses (which reference literals or global constants) are only resolved if
/// a map of the value of each variable is provided. The value of these variable is currently
/// supplied as a `double` as this allows us to represent `int`s as well as `bool`s.
///
/// @param expr            Expression to evaluate
/// @param result          Result of the evaulation if successful, unmodified otherwise
/// @param variableMap     Map of variable name to respective value. If empty, no variables are
///                        resolved.
/// @returns `true` if evaluation was successful, `false` otherwise
///
/// @ingroup sir
/// @{
extern bool evalExprAsDouble(const std::shared_ptr<Expr>& expr, double& result,
                             const std::unordered_map<std::string, double>& variableMap =
                                 std::unordered_map<std::string, double>());
extern bool evalExprAsInteger(const std::shared_ptr<Expr>& expr, int& result,
                              const std::unordered_map<std::string, double>& variableMap =
                                  std::unordered_map<std::string, double>());
extern bool evalExprAsBoolean(const std::shared_ptr<Expr>& expr, bool& result,
                              const std::unordered_map<std::string, double>& variableMap =
                                  std::unordered_map<std::string, double>());
/// @}

//
// TODO refactor_AST: this is TEMPORARY, will be removed in the future
//
using ASTHelper = ast::ASTHelper;
} // namespace sir
} // namespace dawn
