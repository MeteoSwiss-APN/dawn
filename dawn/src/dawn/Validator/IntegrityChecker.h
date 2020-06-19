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

#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTUtil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/Exception.h"

namespace dawn {

//===------------------------------------------------------------------------------------------===//
//     IntegrityChecker
//===------------------------------------------------------------------------------------------===//
/// @brief Perform basic integrity checks on the AST.
class IntegrityChecker : public ast::ASTVisitorForwarding {
  iir::StencilInstantiation* instantiation_;
  iir::StencilMetaInformation& metadata_;

public:
  IntegrityChecker(iir::StencilInstantiation* instantiation);

  void run();

  void visit(const std::shared_ptr<iir::BlockStmt>& stmt) override;
  void visit(const std::shared_ptr<iir::ExprStmt>& stmt) override;
  void visit(const std::shared_ptr<iir::ReturnStmt>& stmt) override;
  void visit(const std::shared_ptr<iir::IfStmt>& stmt) override;
  void visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) override;
  void visit(const std::shared_ptr<iir::AssignmentExpr>& expr) override;
  void visit(const std::shared_ptr<iir::UnaryOperator>& expr) override;
  void visit(const std::shared_ptr<iir::ReductionOverNeighborExpr>& expr) override;
  void visit(const std::shared_ptr<iir::BinaryOperator>& expr) override;
  void visit(const std::shared_ptr<iir::TernaryOperator>& expr) override;
  void visit(const std::shared_ptr<iir::FunCallExpr>& expr) override;

private:
  void iterate(iir::StencilInstantiation* instantiation);
  void iterate(const std::unique_ptr<iir::Stencil>& stencil);
  void iterate(const std::unique_ptr<iir::MultiStage>& multiStage);
  void iterate(const std::unique_ptr<iir::Stage>& stage);
  void iterate(const std::unique_ptr<iir::DoMethod>& doMethod);
};

} // namespace dawn
