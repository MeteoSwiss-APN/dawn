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

namespace dawn {

CompileError::CompileError(const std::string& message, const std::string& file, int line,
                           int column)
    : file_(file), line_(line), column_(column) {
  message_ = message;
  if(!file_.empty()) {
    message_ += " in file '" + file_ + "'";
  }
  if(line_ >= 0) {
    message_ += " at line " + std::to_string(line_);
  }
  if(column_ >= 0) {
    message_ += " at column " + std::to_string(column_);
  }
}

std::string CompileError::getMessage() const { return message_; }
std::string CompileError::getFile() const { return file_; }

int CompileError::getLine() const { return line_; }

const char* CompileError::what() const throw() { return message_.c_str(); }

} // namespace dawn
