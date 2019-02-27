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

#ifndef DAWN_CODEGEN_STENCILFUNCTIONASBCGENERATOR_H
#define DAWN_CODEGEN_STENCILFUNCTIONASBCGENERATOR_H

#include "dawn/CodeGen/ASTCodeGenCXX.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include <memory>

namespace dawn {
namespace codegen {

/// @brief The StencilFunctionAsBCGenerator class parses a stencil function that is used as a
/// boundary
/// condition into it's stringstream. In order to use stencil_functions as boundary conditions, we
/// need them to be members of the stencil-wrapper class. The goal is to template the function s.t
/// every field is a template argument.
class StencilFunctionAsBCGenerator : public ASTCodeGenCXX {
private:
  std::shared_ptr<sir::StencilFunction> function_;
  const std::shared_ptr<iir::StencilInstantiation> instantiation_;

public:
  using Base = ASTCodeGenCXX;
  StencilFunctionAsBCGenerator(
      const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation,
      const std::shared_ptr<sir::StencilFunction>& functionToAnalyze)
      : function_(functionToAnalyze), instantiation_(stencilInstantiation) {}

  void visit(const std::shared_ptr<FieldAccessExpr>& expr);

  inline void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) {
    DAWN_ASSERT_MSG(0, "VerticalRegionDeclStmt not allowed in this context");
  }
  inline void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) {
    DAWN_ASSERT_MSG(0, "StencilCallDeclStmt not allowed in this context");
  }
  inline void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) {
    DAWN_ASSERT_MSG(0, "BoundaryConditionDeclStmt not allowed in this context");
  }
  inline void visit(const std::shared_ptr<StencilFunCallExpr>& expr) {
    DAWN_ASSERT_MSG(0, "StencilFunCallExpr not allowed in this context");
  }
  inline void visit(const std::shared_ptr<StencilFunArgExpr>& expr) {
    DAWN_ASSERT_MSG(0, "StencilFunArgExpr not allowed in this context");
  }

  inline void visit(const std::shared_ptr<ReturnStmt>& stmt) {
    DAWN_ASSERT_MSG(0, "ReturnStmt not allowed in this context");
  }

  void visit(const std::shared_ptr<VarAccessExpr>& expr);

  inline std::string getName(const std::shared_ptr<Stmt>& stmt) const {
    return instantiation_->getFieldNameFromAccessID(instantiation_->getAccessIDFromStmt(stmt));
  }

  inline std::string getName(const std::shared_ptr<Expr>& expr) const {
    return instantiation_->getFieldNameFromAccessID(instantiation_->getAccessIDFromExpr(expr));
  }
};

class BCGenerator {
  const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation_;
  std::stringstream& ss_;

public:
  BCGenerator(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
              std::stringstream& ss)
      : stencilInstantiation_(stencilInstantiation), ss_(ss) {}

  void generate(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt);
};
} // namespace codegen
} // namespace dawn
#endif
