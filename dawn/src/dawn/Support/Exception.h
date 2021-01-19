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

#include "dawn/Support/SourceLocation.h"
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
  int line_;
  int column_;

public:
  CompileError(const std::string& message, const std::string& file = "", int line = -1,
               int column = -1);
  virtual ~CompileError() {}
  std::string getMessage() const;
  std::string getFile() const;
  int getLine() const;
  const char* what() const throw();
};

struct SemanticError : public CompileError {
  SemanticError(const std::string& message = "Semantic Error", const std::string& file = "",
                int line = -1, int column = -1)
      : CompileError(message, file, line, column) {}
  SemanticError(const std::string& message, const std::string& file, SourceLocation loc)
      : CompileError(message, file, loc.Line, loc.Column) {}
};

struct SyntacticError : public CompileError {
  SyntacticError(const std::string& message = "Syntactic Error", const std::string& file = "",
                 int line = -1, int column = -1)
      : CompileError(message, file, line, column) {}
  SyntacticError(const std::string& message, const std::string& file, SourceLocation loc)
      : CompileError(message, file, loc.Line, loc.Column) {}
};

struct LogicError : public CompileError {
  LogicError(const std::string& message = "Logic Error", const std::string& file = "", int line = 0,
             int column = -1)
      : CompileError(message, file, line, column) {}
  LogicError(const std::string& message, const std::string& file, SourceLocation loc)
      : CompileError(message, file, loc.Line, loc.Column) {}
};

} // namespace dawn
