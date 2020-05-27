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

#include "dawn/CodeGen/ASTCodeGenCXX.h"
#include "dawn/CodeGen/CXXUtil.h"
#include <memory>

namespace dawn {
namespace iir {
class StencilInstantiation;
class StencilMetaInformation;
} // namespace iir
namespace sir {
struct StencilFunction;
}
namespace codegen {

/// @brief The StencilFunctionAsBCGenerator class parses a stencil function that is used as a
/// boundary
/// condition into it's stringstream. In order to use stencil_functions as boundary conditions, we
/// need them to be members of the stencil-wrapper class. The goal is to template the function s.t
/// every field is a template argument.
class StencilFunctionAsBCGenerator : public ASTCodeGenCXX {
private:
  std::shared_ptr<sir::StencilFunction> function_;
  const iir::StencilMetaInformation& metadata_;

public:
  using Base = ASTCodeGenCXX;
  using Base::visit;

  StencilFunctionAsBCGenerator(const iir::StencilMetaInformation& metadata,
                               const std::shared_ptr<sir::StencilFunction>& functionToAnalyze)
      : function_(functionToAnalyze), metadata_(metadata) {}

  void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr);

  inline void visit(const std::shared_ptr<iir::VerticalRegionDeclStmt>& stmt) {
    DAWN_ASSERT_MSG(0, "VerticalRegionDeclStmt not allowed in this context");
  }
  inline void visit(const std::shared_ptr<iir::StencilCallDeclStmt>& stmt) {
    DAWN_ASSERT_MSG(0, "StencilCallDeclStmt not allowed in this context");
  }
  inline void visit(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt) {
    DAWN_ASSERT_MSG(0, "BoundaryConditionDeclStmt not allowed in this context");
  }
  inline void visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) {
    DAWN_ASSERT_MSG(0, "StencilFunCallExpr not allowed in this context");
  }
  inline void visit(const std::shared_ptr<iir::StencilFunArgExpr>& expr) {
    DAWN_ASSERT_MSG(0, "StencilFunArgExpr not allowed in this context");
  }

  inline void visit(const std::shared_ptr<iir::ReturnStmt>& stmt) {
    DAWN_ASSERT_MSG(0, "ReturnStmt not allowed in this context");
  }
  inline void visit(const std::shared_ptr<iir::ReductionOverNeighborExpr>& expr) {
    DAWN_ASSERT_MSG(0, "ReductionOverNeighborExpr not allowed in this context");
  }

  void visit(const std::shared_ptr<iir::VarAccessExpr>& expr);

  std::string getName(const std::shared_ptr<iir::VarDeclStmt>& stmt) const;
  std::string getName(const std::shared_ptr<iir::Expr>& expr) const;
};

class BCGenerator {
  const iir::StencilMetaInformation& metadata_;
  std::stringstream& ss_;

public:
  BCGenerator(const iir::StencilMetaInformation& metadata, std::stringstream& ss)
      : metadata_(metadata), ss_(ss) {}

  void generate(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt);
};
} // namespace codegen
} // namespace dawn
