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

#ifndef GSL_SIR_ASTSTRINGIFER_H
#define GSL_SIR_ASTSTRINGIFER_H

#include "gsl/SIR/ASTFwd.h"
#include <iosfwd>
#include <memory>
#include <string>

namespace gsl {

/// @brief Pretty print the AST / ASTNodes using a C-like representation
///
/// This class is merely for debugging.
///
/// @ingroup sir
struct ASTStringifer {
  static std::string toString(const std::shared_ptr<Stmt>& stmt, int initialIndent = 0,
                              bool newLines = true);
  static std::string toString(const std::shared_ptr<Expr>& expr, int initialIndent = 0,
                              bool newLines = true);
  static std::string toString(const AST& ast, int initialIndent = 0, bool newLines = true);
};

/// @name Stream operators
/// @ingroup sir
/// @{
extern std::ostream& operator<<(std::ostream& os, const AST& ast);
extern std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Stmt>& expr);
extern std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Expr>& stmt);
/// @}

} // namespace gsl

#endif
