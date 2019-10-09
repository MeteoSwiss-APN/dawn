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

#ifndef GTCLANG_FRONTEND_GLOBALVARIABLEPARSER_H
#define GTCLANG_FRONTEND_GLOBALVARIABLEPARSER_H

#include "dawn/SIR/SIR.h"
#include "dawn/Support/Json.h"
#include "dawn/Support/NonCopyable.h"
#include "clang/AST/ASTFwd.h"
#include <unordered_map>

namespace gtclang {

class GTClangContext;

/// @brief Convert AST declaration of a stencil to SIR
/// @ingroup frontend
class GlobalVariableParser : dawn::NonCopyable {
  GTClangContext* context_;
  std::shared_ptr<dawn::sir::GlobalVariableMap> variableMap_;
  std::shared_ptr<dawn::json::json> configFile_;
  clang::CXXRecordDecl* recordDecl_;

public:
  GlobalVariableParser(GTClangContext* context);

  /// @brief Get the parsed global variable map
  const std::shared_ptr<dawn::sir::GlobalVariableMap>& getGlobalVariableMap() const;

  /// @brief Check if global variable exists
  bool has(const std::string& name) const;

  /// @brief Parse the global struct
  void parseGlobals(clang::CXXRecordDecl* recordDecl);

  /// @brief Get the CXXRecordDecl of the globals (may return NULL)
  clang::CXXRecordDecl* getRecordDecl() const;

  /// @brief Check if a `globals` was parsed
  bool isEmpty() const { return getRecordDecl() == nullptr; }
};

} // namespace gtclang

#endif
