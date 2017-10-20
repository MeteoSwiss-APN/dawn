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

namespace gtclang {

//template <typename... Args>
//void ParsedString::argumentParsing(const std::shared_ptr<dawn::Expr>& argument, Args&&... args) {
//  argumentParsingImpl(argument);
//  argumentParsing(std::forward<Args>(args)...);
//}

//void ParsedString::argumentParsingImpl(const std::shared_ptr<dawn::Expr>& argument) {
//  if(dawn::VarAccessExpr* expr = dawn::dyn_cast<dawn::VarAccessExpr>(argument.get())) {
//    addVariable(expr->getName());
//  } else if(dawn::FieldAccessExpr* expr = dawn::dyn_cast<dawn::FieldAccessExpr>(argument.get())) {
//    addField(expr->getName());
//  } else {
//    dawn_unreachable("invalid expression");
//  }
//}

void ParsedString::dump() {
  std::cout << "function call: " << std::endl;
  std::cout << functionCall_ << std::endl;
  std::cout << "all fields: " << dawn::RangeToString()(fields_) << std::endl;
  std::cout << "all variables: " << dawn::RangeToString()(variables_) << std::endl;
}



} // namespace gtclang
