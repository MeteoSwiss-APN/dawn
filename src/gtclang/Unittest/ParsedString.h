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

#ifndef GTCLANG_UNITTEST_PARSEDSTRING_H
#define GTCLANG_UNITTEST_PARSEDSTRING_H
#include "dawn/SIR/ASTExpr.h"
#include "dawn/SIR/ASTStmt.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Array.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/Unreachable.h"
#include <string>
#include <vector>

namespace gtclang {

/// @brief parsing unittest-strings to automatically write simple stencils
/// @ingroup unittest
/// The class is filled with parse
/// To generate the stencil, we call FileWriter with this class as an argument
class ParsedString {
public:
  ParsedString() = default;
  ParsedString(ParsedString&&) = default;
  ParsedString(const ParsedString&) = default;
  ParsedString(const std::string& functionCall);

  /// @brief get all the parsed Fields
  const std::vector<std::string>& getFields() const;

  /// @brief get all the parsed Variables
  const std::vector<std::string>& getVariables() const;

  /// @brief retruns the function call that was to pe parsed
  const std::string& getCall() const { return functionCall_; }

  /// @brief recursive argument parsing to read all the fields given to specify the function call
  /// @{
  template <typename... Args>
  void argumentParsing(const std::shared_ptr<dawn::Expr>& argument, Args&&... args) {
    argumentParsingImpl(argument);
    argumentParsing(std::forward<Args>(args)...);
  }
  void argumentParsing();
  /// @}

  /// @brief dumps the call with all its fields and variables to std::cout
  void dump();

private:
  /// @brief lets the implementation add fields to the local storage
  void addField(const std::string& field) { fields_.push_back(field); }

  /// @brief lets the implementation add variables to the local storage
  void addVariable(const std::string& variable) { variables_.push_back(variable); }

  /// @brief recursive argument parsing to read all the fields given to specify the function call
  void argumentParsingImpl(const std::shared_ptr<dawn::Expr>& argument);

  std::vector<std::string> fields_;
  std::vector<std::string> variables_;
  std::string functionCall_;
};

/// @brief parses a string describing an operation with its respective variables
/// @ingroup unittest
/// @param[in] Function call as a string (e.g "a = b + c")
/// @param[in] Declaration of Variables as Fields or variable accesses [dawn::field("a"),
/// dawn::var("b")]
/// @param[out] An object containing all the information to autogenerate the corresponding stencil
/// to a File
template <typename... Args>
ParsedString parse(const std::string& functionCall, Args&&... args) {
  ParsedString parsed(functionCall);
  parsed.argumentParsing(std::forward<Args>(args)...);
  return parsed;
}
} // namespace gtclang

#endif
