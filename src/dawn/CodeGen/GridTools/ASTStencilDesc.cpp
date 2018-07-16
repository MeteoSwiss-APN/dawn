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

#include "dawn/CodeGen/GridTools/ASTStencilDesc.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/AST.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {
namespace codegen {
namespace gt {

ASTStencilDesc::ASTStencilDesc(
    const std::shared_ptr<iir::StencilInstantiation> instantiation,
    const std::unordered_map<int, std::vector<std::string>>& StencilIDToStencilNameMap,
    const std::unordered_map<int, std::string>& stencilIdToArguments)
    : ASTCodeGenCXX(), instantiation_(instantiation),
      StencilIDToStencilNameMap_(StencilIDToStencilNameMap),
      stencilIdToArguments_(stencilIdToArguments) {}

ASTStencilDesc::~ASTStencilDesc() {}

std::string ASTStencilDesc::getName(const std::shared_ptr<Stmt>& stmt) const {
  return instantiation_->getNameFromAccessID(instantiation_->getAccessIDFromStmt(stmt));
}

std::string ASTStencilDesc::getName(const std::shared_ptr<Expr>& expr) const {
  return instantiation_->getNameFromAccessID(instantiation_->getAccessIDFromExpr(expr));
}

//===------------------------------------------------------------------------------------------===//
//     Stmt
//===------------------------------------------------------------------------------------------===//

void ASTStencilDesc::visit(const std::shared_ptr<BlockStmt>& stmt) { Base::visit(stmt); }

void ASTStencilDesc::visit(const std::shared_ptr<ExprStmt>& stmt) { Base::visit(stmt); }

void ASTStencilDesc::visit(const std::shared_ptr<ReturnStmt>& stmt) {
  dawn_unreachable("ReturnStmt not allowed in StencilDesc AST");
}

void ASTStencilDesc::visit(const std::shared_ptr<VarDeclStmt>& stmt) { Base::visit(stmt); }

void ASTStencilDesc::visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) {
  dawn_unreachable("VerticalRegionDeclStmt not allowed in StencilDesc AST");
}

void ASTStencilDesc::visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) {
  int StencilID = instantiation_->getStencilCallToStencilIDMap().find(stmt)->second;

  for(const std::string& stencilName : StencilIDToStencilNameMap_.find(StencilID)->second) {
    ss_ << std::string(indent_, ' ') << stencilName << ".get_stencil().run();\n";
  }
}

void ASTStencilDesc::visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) {
  Extents extents = instantiation_->getBoundaryConditionExtentsFromBCStmt(stmt);
  int haloIMinus = abs(extents[0].Minus);
  int haloIPlus = abs(extents[0].Plus);
  int haloJMinus = abs(extents[1].Minus);
  int haloJPlus = abs(extents[1].Plus);
  int haloKMinus = abs(extents[2].Minus);
  int haloKPlus = abs(extents[2].Plus);
  std::string fieldname = stmt->getFields()[0]->Name;

  // Set up the halos
  std::string halosetup = dawn::format(
      "gridtools::array< gridtools::halo_descriptor, 3 > halos;\n"
      "halos[0] =gridtools::halo_descriptor(%i, %i, "
      "%s.get_storage_info_ptr()->template begin<0>(),%s.get_storage_info_ptr()->template "
      "end<0>(), "
      "%s.get_storage_info_ptr()->template total_length<0>());\nhalos[1] = "
      "gridtools::halo_descriptor(%i, "
      "%i, "
      "%s.get_storage_info_ptr()->template begin<1>(),%s.get_storage_info_ptr()->template "
      "end<1>(), "
      "%s.get_storage_info_ptr()->template total_length<1>());\nhalos[2] = "
      "gridtools::halo_descriptor(%i, "
      "%i, "
      "%s.get_storage_info_ptr()->template begin<2>(),%s.get_storage_info_ptr()->template "
      "end<2>(), "
      "%s.get_storage_info_ptr()->template total_length<2>());\n",
      haloIMinus, haloIPlus, fieldname, fieldname, fieldname, haloJMinus, haloJPlus, fieldname,
      fieldname, fieldname, haloKMinus, haloKPlus, fieldname, fieldname, fieldname);
  std::string makeView = "";

  // Create the views for the fields
  for(int i = 0; i < stmt->getFields().size(); ++i) {
    auto fieldName = stmt->getFields()[i]->Name;
    makeView +=
        dawn::format("auto %s_view = GT_BACKEND_DECISION_viewmaker(%s);\n", fieldName, fieldName);
  }
  std::string bcapply = "GT_BACKEND_DECISION_bcapply<" + stmt->getFunctor() + " >(halos, " +
                        stmt->getFunctor() + "()).apply(";
  for(int i = 0; i < stmt->getFields().size(); ++i) {
    bcapply += stmt->getFields()[i]->Name + "_view";
    if(i < stmt->getFields().size() - 1) {
      bcapply += ", ";
    }
  }
  bcapply += ");\n";

  ss_ << halosetup;
  ss_ << makeView;
  ss_ << bcapply;
}

void ASTStencilDesc::visit(const std::shared_ptr<IfStmt>& stmt) { Base::visit(stmt); }

//===------------------------------------------------------------------------------------------===//
//     Expr
//===------------------------------------------------------------------------------------------===//

void ASTStencilDesc::visit(const std::shared_ptr<UnaryOperator>& expr) { Base::visit(expr); }

void ASTStencilDesc::visit(const std::shared_ptr<BinaryOperator>& expr) { Base::visit(expr); }

void ASTStencilDesc::visit(const std::shared_ptr<AssignmentExpr>& expr) { Base::visit(expr); }

void ASTStencilDesc::visit(const std::shared_ptr<TernaryOperator>& expr) { Base::visit(expr); }

void ASTStencilDesc::visit(const std::shared_ptr<FunCallExpr>& expr) { Base::visit(expr); }

void ASTStencilDesc::visit(const std::shared_ptr<StencilFunCallExpr>& expr) {
  dawn_unreachable("StencilFunCallExpr not allowed in StencilDesc AST");
}

void ASTStencilDesc::visit(const std::shared_ptr<StencilFunArgExpr>& expr) {
  dawn_unreachable("StencilFunArgExpr not allowed in StencilDesc AST");
}

void ASTStencilDesc::visit(const std::shared_ptr<VarAccessExpr>& expr) {
  if(instantiation_->isGlobalVariable(instantiation_->getAccessIDFromExpr(expr)))
    ss_ << "globals::get().";

  ss_ << getName(expr);

  if(expr->isArrayAccess()) {
    ss_ << "[";
    expr->getIndex()->accept(*this);
    ss_ << "]";
  }
}

void ASTStencilDesc::visit(const std::shared_ptr<LiteralAccessExpr>& expr) { Base::visit(expr); }

void ASTStencilDesc::visit(const std::shared_ptr<FieldAccessExpr>& expr) {
  dawn_unreachable("FieldAccessExpr not allowed in StencilDesc AST");
}

} // namespace gt
} // namespace codegen
} // namespace dawn
