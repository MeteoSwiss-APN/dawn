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

#include "dawn/AST/ASTUtil.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/Exception.h"

namespace dawn {

//===------------------------------------------------------------------------------------------===//
//     IntegrityChecker
//===------------------------------------------------------------------------------------------===//
/// @brief Perform basic integrity checks on the AST.
class IntegrityChecker : public ast::ASTVisitorForwardingNonConst {
  iir::StencilInstantiation* instantiation_;
  iir::StencilMetaInformation& metadata_;
  // are we in a loop or reduction expression?
  bool parentHasIterationContext_ = false;
  // current field dimensions on stack
  int curDimensions_ = -1;

public:
  IntegrityChecker(iir::StencilInstantiation* instantiation);

  void run();

  void visit(const std::shared_ptr<ast::VarAccessExpr>& stmt) override;
  void visit(const std::shared_ptr<ast::AssignmentExpr>& expr) override;
  void visit(const std::shared_ptr<ast::FieldAccessExpr>& expr) override;
  void visit(const std::shared_ptr<ast::ReductionOverNeighborExpr>& expr) override;
  void visit(const std::shared_ptr<ast::LoopStmt>& expr) override;

private:
  void iterate(iir::StencilInstantiation* instantiation);
  void iterate(const std::unique_ptr<iir::Stencil>& stencil);
  void iterate(const std::unique_ptr<iir::MultiStage>& multiStage);
  void iterate(const std::unique_ptr<iir::Stage>& stage);
  void iterate(const std::unique_ptr<iir::DoMethod>& doMethod);
};

} // namespace dawn
