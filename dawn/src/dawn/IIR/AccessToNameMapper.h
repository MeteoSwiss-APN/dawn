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

#include "dawn/AST/ASTVisitor.h"
#include <stack>
#include <string>
#include <unordered_map>

namespace dawn {
namespace iir {

class StencilFunctionInstantiation;
class StencilMetaInformation;

/// @brief Dump AST to string
class AccessToNameMapper : public ast::ASTVisitorForwardingNonConst {
  const StencilMetaInformation& metaData_;
  std::stack<StencilFunctionInstantiation*> curFunctionInstantiation_;

  // this is a map from accessID to name of all accesses within the AST,
  // It is computed here since the accessIDs can be stored in different stencil/function
  // instantations
  std::unordered_map<int, std::string> accessIDToName_;

public:
  AccessToNameMapper(const StencilMetaInformation& metaData) : metaData_(metaData) {}

  virtual void visit(const std::shared_ptr<ast::VarDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<ast::StencilFunCallExpr>& expr) override;
  virtual void visit(const std::shared_ptr<ast::VarAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<ast::FieldAccessExpr>& expr) override;

  std::string getNameFromAccessID(int accessID) const { return accessIDToName_.at(accessID); }
  bool hasAccessID(int accessID) const { return accessIDToName_.count(accessID); }

private:
  void insertAccessInfo(const std::shared_ptr<ast::Expr>& expr);
  void insertAccessInfo(const std::shared_ptr<ast::VarDeclStmt>& stmt);
};

} // namespace iir
} // namespace dawn
