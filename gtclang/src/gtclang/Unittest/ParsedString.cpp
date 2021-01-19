//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _
//                        | |     | |
//                    __ _| |_ ___| | __ _ _ __   __ _
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gtclang/Unittest/ParsedString.h"
#include "dawn/Support/StringUtil.h"

#include <iostream>

namespace gtclang {

ParsedString::ParsedString(const std::string& functionCall) : functionCall_(functionCall) {
  if(functionCall_.back() == ';') {
    functionCall_.pop_back();
  }
}

const std::vector<std::string>& ParsedString::getFields() const { return fields_; }

const std::vector<std::string>& ParsedString::getVariables() const { return variables_; }

void ParsedString::argumentParsing() {}

void ParsedString::dump() {
  std::cout << "function call: " << std::endl;
  std::cout << functionCall_ << std::endl;
  std::cout << "all fields: " << dawn::RangeToString()(fields_) << std::endl;
  std::cout << "all variables: " << dawn::RangeToString()(variables_) << std::endl;
}

void ParsedString::argumentParsingImpl(const std::shared_ptr<dawn::ast::Expr>& argument) {
  if(dawn::ast::VarAccessExpr* expr = dawn::dyn_cast<dawn::ast::VarAccessExpr>(argument.get())) {
    addVariable(expr->getName());
  } else if(dawn::ast::FieldAccessExpr* expr =
                dawn::dyn_cast<dawn::ast::FieldAccessExpr>(argument.get())) {
    addField(expr->getName());
  } else {
    dawn_unreachable("invalid expression");
  }
}

} // namespace gtclang
