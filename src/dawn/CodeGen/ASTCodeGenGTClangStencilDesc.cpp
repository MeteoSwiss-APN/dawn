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

#include "dawn/CodeGen/ASTCodeGenGTClangStencilDesc.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/AST.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {

static Extents analyzestencilExtents(const std::shared_ptr<Stencil>& s, int ID) {
    std::unordered_map<int, Extents> fieldIDtoFullExtentSize;
      Stencil& stencil = *s;

      int numStages = stencil.getNumStages();

      // loop over stages
      for(int i = 0; i < numStages; ++i) {
        Stage& stage = *(stencil.getStage(i));

        Extents const& stageExtent = stage.getExtents();
        for(auto& field : stage.getFields()) {
          if(fieldIDtoFullExtentSize.count(ID) == 0) {
            fieldIDtoFullExtentSize.emplace(ID, field.Extent);
            fieldIDtoFullExtentSize.find(ID)->second.add(stageExtent);
          } else {
            fieldIDtoFullExtentSize.find(ID)->second.merge(field.Extent);
            fieldIDtoFullExtentSize.find(ID)->second.add(stageExtent);
          }
        }
      }

    return fieldIDtoFullExtentSize.find(ID)->second;
}

ASTCodeGenGTClangStencilDesc::ASTCodeGenGTClangStencilDesc(
    const StencilInstantiation* instantiation,
    const std::unordered_map<int, std::vector<std::string>>& StencilIDToStencilNameMap)
    : ASTCodeGenCXX(), instantiation_(instantiation),
      StencilIDToStencilNameMap_(StencilIDToStencilNameMap) {}

ASTCodeGenGTClangStencilDesc::~ASTCodeGenGTClangStencilDesc() {}

const std::string& ASTCodeGenGTClangStencilDesc::getName(const std::shared_ptr<Stmt>& stmt) const {
  return instantiation_->getNameFromAccessID(instantiation_->getAccessIDFromStmt(stmt));
}

const std::string& ASTCodeGenGTClangStencilDesc::getName(const std::shared_ptr<Expr>& expr) const {
  return instantiation_->getNameFromAccessID(instantiation_->getAccessIDFromExpr(expr));
}

//===------------------------------------------------------------------------------------------===//
//     Stmt
//===------------------------------------------------------------------------------------------===//

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<BlockStmt>& stmt) {
  Base::visit(stmt);
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<ExprStmt>& stmt) {
  Base::visit(stmt);
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<ReturnStmt>& stmt) {
  dawn_unreachable("ReturnStmt not allowed in StencilDesc AST");
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<VarDeclStmt>& stmt) {
  Base::visit(stmt);
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) {
  dawn_unreachable("VerticalRegionDeclStmt not allowed in StencilDesc AST");
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) {
  int StencilID = instantiation_->getStencilCallToStencilIDMap().find(stmt)->second;
  StencilIDsVisited_.push_back(StencilID);
  for(const std::string& stencilName : StencilIDToStencilNameMap_.find(StencilID)->second)
    ss_ << std::string(indent_, ' ') << stencilName << ".get_stencil()->run();\n";
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) {

  auto calculateHaloExtents = [&](std::string fieldname) {

    Extents fullExtent;
    // Did we already apply a BoundaryCondition for this field?
    // This is the first time we apply a BC to this field, we traverse all stencils that were
    // applied before
    if(StencilBCsApplied_.count(fieldname) == 0) {
      for(auto& stencil : instantiation_->getStencils()) {
        for(const int& ID : StencilIDsVisited_) {
          if(ID == stencil->getStencilID()) {
            fullExtent.merge(
                analyzestencilExtents(stencil, instantiation_->getAccessIDFromName(fieldname)));
            if(StencilBCsApplied_.count(fieldname) == 0) {
              StencilBCsApplied_.emplace(fieldname, std::vector<int>{stencil->getStencilID()});
            } else {
              StencilBCsApplied_.find(fieldname)->second.push_back(stencil->getStencilID());
            }
            break;
          }
        }
      }
    }
    // This Field already had BC's applied. We remove all those stencils from the compuattion
    else {
      std::vector<int> stencilIDsToVisit(StencilIDsVisited_);
      for(int traveresedID : StencilBCsApplied_.find(fieldname)->second) {
        stencilIDsToVisit.erase(
            std::remove(stencilIDsToVisit.begin(), stencilIDsToVisit.end(), traveresedID),
            stencilIDsToVisit.end());

        for(const auto& stencil : instantiation_->getStencils()) {
          for(const int& ID : stencilIDsToVisit) {
            if(ID == stencil->getStencilID()) {
              fullExtent.merge(
                  analyzestencilExtents(stencil, instantiation_->getAccessIDFromName(fieldname)));
              StencilBCsApplied_.find(fieldname)->second.push_back(stencil->getStencilID());
              break;
            }
          }
        }
      }
    }
    return fullExtent;
  };

  // we need to calculate the extents in which the field needs a boundary condition
  Extents extents = calculateHaloExtents(stmt->getFields()[0]->Name);
  int haloIMinus = extents[0].Minus;
  int haloIPlus = extents[0].Plus;
  int haloJMinus = extents[1].Minus;
  int haloJPlus = extents[1].Plus;
  int haloKMinus = extents[2].Minus;
  int haloKPlus = extents[2].Plus;
  std::string fieldname = stmt->getFields()[0]->Name;

  // Set up the halos
  std::string halosetup = dawn::format(
      "gridtools::array< gridtools::halo_descriptor, 3 > halos;\n"
      "halos[0] =gridtools::halo_descriptor(%i, %i, "
      "%s->get_storage_info_ptr()->begin<0>(),%s->get_storage_info_ptr()->end<0>(), "
      "%s->get_storage_info_ptr()->total_length<0>());\nhalos[1] = gridtools::halo_descriptor(%i, "
      "%i, "
      "%s->get_storage_info_ptr()->begin<1>(),%s->get_storage_info_ptr()->end<1>(), "
      "%s->get_storage_info_ptr()->total_length<1>());\nhalos[2] = gridtools::halo_descriptor(%i, "
      "%i, "
      "%s->get_storage_info_ptr()->begin<2>(),%s->get_storage_info_ptr()->end<2>(), "
      "%s->get_storage_info_ptr()->total_length<2>());\n",
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

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<IfStmt>& stmt) { Base::visit(stmt); }

//===------------------------------------------------------------------------------------------===//
//     Expr
//===------------------------------------------------------------------------------------------===//

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<UnaryOperator>& expr) {
  Base::visit(expr);
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<BinaryOperator>& expr) {
  Base::visit(expr);
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<AssignmentExpr>& expr) {
  Base::visit(expr);
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<TernaryOperator>& expr) {
  Base::visit(expr);
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<FunCallExpr>& expr) {
  Base::visit(expr);
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<StencilFunCallExpr>& expr) {
  dawn_unreachable("StencilFunCallExpr not allowed in StencilDesc AST");
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<StencilFunArgExpr>& expr) {
  dawn_unreachable("StencilFunArgExpr not allowed in StencilDesc AST");
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<VarAccessExpr>& expr) {
  if(instantiation_->isGlobalVariable(instantiation_->getAccessIDFromExpr(expr)))
    ss_ << "globals::get().";

  ss_ << getName(expr);

  if(expr->isArrayAccess()) {
    ss_ << "[";
    expr->getIndex()->accept(*this);
    ss_ << "]";
  }
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<LiteralAccessExpr>& expr) {
  Base::visit(expr);
}

void ASTCodeGenGTClangStencilDesc::visit(const std::shared_ptr<FieldAccessExpr>& expr) {
  dawn_unreachable("FieldAccessExpr not allowed in StencilDesc AST");
}

} // namespace dawn
