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

#include "dawn/Support/NonCopyable.h"
#include <string>

namespace dawn {
namespace iir {

class InstantiationHelper : NonCopyable {

public:
  /// @brief Generate a unique name for a local variable
  static std::string makeLocalVariablename(const std::string& name, int AccessID);

  /// @brief Generate a unique name for a temporary field
  static std::string makeTemporaryFieldname(const std::string& name, int AccessID);

  /// @brief Extract the name of a local variable
  ///
  /// Reverse the effect of `makeLocalVariablename`.
  static std::string extractLocalVariablename(const std::string& name);

  /// @brief Extract the name of a local variable
  ///
  /// Reverse the effect of `makeTemporaryFieldname`.
  static std::string extractTemporaryFieldname(const std::string& name);

  /// @brief Name used for all `StencilCallDeclStmt` in the stencil description AST
  /// (`getStencilDescStatements`) to signal code-gen it should insert a call to the gridtools
  /// stencil here
  static std::string makeStencilCallCodeGenName(int StencilID);

  /// @brief Check if the given name of a `StencilCallDeclStmt` was generate by
  /// `makeStencilCallCodeGenName`
  static bool isStencilCallCodeGenName(const std::string& name);
};
} // namespace iir
} // namespace dawn
