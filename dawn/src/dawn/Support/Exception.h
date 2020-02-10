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

#ifndef DAWN_SUPPORT_EXCEPTION_H
#define DAWN_SUPPORT_EXCEPTION_H

#include <exception>
#include <string>

namespace dawn {

// The four categories of compiler errors:
//   1) Lexical: misspellings of identifiers, keywords or operators, etc.
//   2) Syntactical: missing semicolon or unbalanced parenthesis, etc.
//   3) Semantical: incompatible value assignments or type mismatches, etc.
//   4) Logical: unreachable code, infinite loops, etc.

class CompileError : public std::exception {
  std::string message_;
  std::string file_;
  unsigned line_;

public:
  CompileError(const std::string& message, const std::string& file = "", unsigned line = 0);
  virtual ~CompileError() {}
  std::string getMessage() const;
  std::string getFile() const;
  unsigned getLine() const;
  const char* what() const throw();
};

struct SemanticError : public CompileError {
  SemanticError(const std::string& message = "Semantic Error", const std::string& file = "",
                unsigned line = 0);
};

} // namespace dawn

#endif // DAWN_SUPPORT_EXCEPTION_H
