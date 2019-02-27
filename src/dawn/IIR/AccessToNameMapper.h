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

#ifndef DAWN_IIR_ACCESSTONAMEMAPPER_H
#define DAWN_IIR_ACCESSTONAMEMAPPER_H

#include "dawn/SIR/ASTVisitor.h"
#include <stack>
#include <unordered_map>

namespace dawn {
namespace iir {

class StencilInstantiation;
class StencilFunctionInstantiation;

/// @brief Dump AST to string
class AccessToNameMapper : public ASTVisitorForwarding {
  int scopeDepth_;
  const iir::StencilInstantiation* stencilInstantiation_;
  std::stack<iir::StencilFunctionInstantiation*> curFunctionInstantiation_;

  // this is a map from accessID to name of all accesses within the AST,
  // It is computed here since the accessIDs can be stored in different stencil/function
  // instantations
  std::unordered_map<int, std::string> accessIDToName_;

public:
  AccessToNameMapper(const iir::StencilInstantiation* instantiation)
      : scopeDepth_(0), stencilInstantiation_(instantiation) {}

  virtual void visit(const std::shared_ptr<VarDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override;
  virtual void visit(const std::shared_ptr<VarAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override;

  std::string getNameFromAccessID(int accessID) const { return accessIDToName_.at(accessID); }
  bool hasAccessID(int accessID) const { return accessIDToName_.count(accessID); }

private:
  void insertAccessInfo(const std::shared_ptr<Expr>& expr);
  void insertAccessInfo(const std::shared_ptr<Stmt>& stmt);
};

} // namespace iir
} // namespace dawn

#endif
