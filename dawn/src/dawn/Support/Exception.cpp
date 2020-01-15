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

#include "dawn/Support/Exception.h"
#include <cstring>

namespace dawn {

CompileError::CompileError(const std::string& message, const std::string& file, unsigned line)
    : message_(message), file_(file), line_(line) {}

std::string CompileError::getMessage() const {
  std::string message = message_;
  if(!file_.empty()) {
    message += " in file '" + file_ + "'";
  }
  if(line_ > 0) {
    message += " at line " + std::to_string(line_);
  }
  return message;
}

std::string CompileError::getFile() const { return file_; }

unsigned CompileError::getLine() const { return line_; }

// const char* CompileException::what() const throw() {}

SemanticError::SemanticError(const std::string& message, const std::string& file, unsigned line)
    : CompileError(message, file, line) {}

} // namespace dawn
