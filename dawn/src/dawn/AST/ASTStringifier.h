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

#include "dawn/AST/ASTFwd.h"
#include <iosfwd>
#include <memory>
#include <string>

namespace dawn {
namespace ast {
/// @brief Pretty print the AST / ASTNodes using a C-like representation
///
/// This class is merely for debugging.
///
/// @ingroup ast
struct ASTStringifier {
  ASTStringifier() = delete;

  static std::string toString(const std::shared_ptr<Stmt>& stmt, int initialIndent = 0,
                              bool newLines = true);
  static std::string toString(const std::shared_ptr<Expr>& expr, int initialIndent = 0,
                              bool newLines = true);
  static std::string toString(const AST& ast, int initialIndent = 0, bool newLines = true);
};

/// @name Stream operators
/// @ingroup ast
/// @{
extern std::ostream& operator<<(std::ostream& os, const AST& ast);
extern std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Stmt>& expr);
extern std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Expr>& stmt);
/// @}
} // namespace ast
} // namespace dawn
