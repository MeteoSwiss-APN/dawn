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

#include "GenerateInMemoryStencils.h"

#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/ASTUtil.h"
#include "dawn/IIR/AccessComputation.h"
#include "dawn/IIR/FieldAccessMetadata.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/InstantiationHelper.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassSetStageName.h"
#include "dawn/Optimizer/PassTemporaryType.h"
#include "dawn/Optimizer/StatementMapper.h"
#include "dawn/SIR/ASTFwd.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Support/Logger.h"
#include "dawn/Support/STLExtras.h"
#include "dawn/Unittest/IIRBuilder.h"

using namespace dawn;

std::shared_ptr<iir::StencilInstantiation> createCopyStencilIIRInMemory(ast::GridType gridType) {
  auto target = std::make_shared<iir::StencilInstantiation>(gridType);

  ///////////////// Generation of the IIR
  sir::Attr attributes;
  int stencilID = target->nextUID();
  target->getIIR()->insertChild(
      std::make_unique<iir::Stencil>(target->getMetaData(), attributes, stencilID),
      target->getIIR());
  const auto& IIRStencil = target->getIIR()->getChild(0);
  // One Multistage with a parallel looporder
  IIRStencil->insertChild(
      std::make_unique<iir::MultiStage>(target->getMetaData(), iir::LoopOrderKind::Parallel));
  const auto& IIRMSS = (IIRStencil)->getChild(0);
  IIRMSS->setID(target->nextUID());

  // Create one stage inside the MSS
  IIRMSS->insertChild(std::make_unique<iir::Stage>(target->getMetaData(), target->nextUID()));
  const auto& IIRStage = IIRMSS->getChild(0);

  // Create one doMethod inside the Stage that spans the full domain
  IIRStage->insertChild(std::make_unique<iir::DoMethod>(
      iir::Interval(sir::Interval{sir::Interval::Start, sir::Interval::End}),
      target->getMetaData()));
  const auto& IIRDoMethod = IIRStage->getChild(0);
  IIRDoMethod->setID(target->nextUID());

  // create the statement
  auto makeFieldDimensions = []() -> sir::FieldDimensions {
    return sir::FieldDimensions(sir::HorizontalFieldDimension(ast::cartesian, {true, true}), true);
  };

  auto sirInField = std::make_shared<sir::Field>("in_field", makeFieldDimensions());
  sirInField->IsTemporary = false;
  auto sirOutField = std::make_shared<sir::Field>("out_field", makeFieldDimensions());
  sirOutField->IsTemporary = false;

  auto lhs = std::make_shared<ast::FieldAccessExpr>(sirOutField->Name);
  lhs->setID(target->nextUID());
  auto rhs = std::make_shared<ast::FieldAccessExpr>(sirInField->Name);
  rhs->setID(target->nextUID());

  int in_fieldID = target->getMetaData().addField(iir::FieldAccessType::APIField, sirInField->Name,
                                                  std::move(sirInField->Dimensions));
  int out_fieldID = target->getMetaData().addField(
      iir::FieldAccessType::APIField, sirOutField->Name, std::move(sirOutField->Dimensions));

  lhs->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(out_fieldID);
  rhs->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(in_fieldID);

  auto expr = std::make_shared<ast::AssignmentExpr>(lhs, rhs);
  expr->setID(target->nextUID());
  auto stmt = iir::makeExprStmt(expr);
  stmt->setID(target->nextUID());

  // Add the accesses:
  iir::Accesses callerAccesses;
  callerAccesses.addWriteExtent(out_fieldID, iir::Extents(ast::cartesian, 0, 0, 0, 0, 0, 0));
  callerAccesses.addReadExtent(in_fieldID, iir::Extents(ast::cartesian, 0, 0, 0, 0, 0, 0));
  stmt->getData<iir::IIRStmtData>().CallerAccesses = std::make_optional(std::move(callerAccesses));
  // And add the statement to it
  IIRDoMethod->getAST().push_back(std::move(stmt));
  IIRDoMethod->updateLevel();

  // Add the control flow descriptor to the IIR
  auto stencilCall = std::make_shared<ast::StencilCall>("generatedDriver");
  stencilCall->Args.push_back(sirInField->Name);
  stencilCall->Args.push_back(sirOutField->Name);
  auto placeholderStencil = std::make_shared<ast::StencilCall>(
      iir::InstantiationHelper::makeStencilCallCodeGenName(stencilID));
  auto stencilCallDeclStmt = iir::makeStencilCallDeclStmt(placeholderStencil);
  // Register the call and set it as a replacement for the next vertical region
  target->getMetaData().addStencilCallStmt(stencilCallDeclStmt, stencilID);
  target->getIIR()->getControlFlowDescriptor().insertStmt(stencilCallDeclStmt);

  ///////////////// Generation of the Metadata

  target->getMetaData().setStencilName("generated");

  for(const auto& MS : iterateIIROver<iir::MultiStage>(*(target->getIIR()))) {
    MS->update(iir::NodeUpdateType::levelAndTreeAbove);
  }
  // Iterate all statements (top -> bottom)
  for(const auto& stagePtr : iterateIIROver<iir::Stage>(*(target->getIIR()))) {
    iir::Stage& stage = *stagePtr;
    for(const auto& doMethod : stage.getChildren()) {
      doMethod->update(iir::NodeUpdateType::level);
    }
    stage.update(iir::NodeUpdateType::level);
  }
  for(const auto& MSPtr : iterateIIROver<iir::Stage>(*(target->getIIR()))) {
    MSPtr->update(iir::NodeUpdateType::levelAndTreeAbove);
  }

  return target;
}

std::shared_ptr<iir::StencilInstantiation> createLapStencilIIRInMemory(ast::GridType gridType) {
  auto target = std::make_shared<iir::StencilInstantiation>(gridType);

  ///////////////// Generation of the IIR
  sir::Attr attributes;
  int stencilID = target->nextUID();
  target->getIIR()->insertChild(
      std::make_unique<iir::Stencil>(target->getMetaData(), attributes, stencilID),
      target->getIIR());
  const auto& IIRStencil = target->getIIR()->getChild(0);
  // One Multistage with a parallel looporder
  IIRStencil->insertChild(
      std::make_unique<iir::MultiStage>(target->getMetaData(), iir::LoopOrderKind::Parallel));
  const auto& IIRMSS = (IIRStencil)->getChild(0);
  IIRMSS->setID(target->nextUID());

  auto IIRStage1 = std::make_unique<iir::Stage>(target->getMetaData(), target->nextUID());
  auto IIRStage2 = std::make_unique<iir::Stage>(target->getMetaData(), target->nextUID());

  IIRStage1->setExtents(iir::Extents(ast::cartesian, -1, +1, -1, +1, 0, 0));

  // Create one doMethod inside the Stage that spans the full domain
  IIRStage1->insertChild(std::make_unique<iir::DoMethod>(
      iir::Interval(sir::Interval{sir::Interval::Start, sir::Interval::End}),
      target->getMetaData()));
  const auto& IIRDoMethod1 = IIRStage1->getChild(0);
  IIRDoMethod1->setID(target->nextUID());

  IIRStage2->insertChild(std::make_unique<iir::DoMethod>(
      iir::Interval(sir::Interval{sir::Interval::Start, sir::Interval::End}),
      target->getMetaData()));
  const auto& IIRDoMethod2 = IIRStage2->getChild(0);
  IIRDoMethod2->setID(target->nextUID());

  // Create two stages inside the MSS
  IIRMSS->insertChild(std::move(IIRStage1));
  IIRMSS->insertChild(std::move(IIRStage2));

  // create the statement
  auto makeFieldDimensions = []() -> sir::FieldDimensions {
    return sir::FieldDimensions(sir::HorizontalFieldDimension(ast::cartesian, {true, true}), true);
  };

  auto sirInField = std::make_shared<sir::Field>("in", makeFieldDimensions());
  sirInField->IsTemporary = false;
  auto sirOutField = std::make_shared<sir::Field>("out", makeFieldDimensions());
  sirOutField->IsTemporary = false;
  auto sirTmpField = std::make_shared<sir::Field>("tmp", makeFieldDimensions());
  sirOutField->IsTemporary = true;

  auto lhsTmp = std::make_shared<ast::FieldAccessExpr>(sirTmpField->Name);
  lhsTmp->setID(target->nextUID());

  auto rhsInT1 = std::make_shared<ast::FieldAccessExpr>(sirInField->Name,
                                                        ast::Offsets{ast::cartesian, 0, -2, 0});
  auto rhsInT2 = std::make_shared<ast::FieldAccessExpr>(sirInField->Name,
                                                        ast::Offsets{ast::cartesian, 0, +2, 0});
  auto rhsInT3 = std::make_shared<ast::FieldAccessExpr>(sirInField->Name,
                                                        ast::Offsets{ast::cartesian, -2, 0, 0});
  auto rhsInT4 = std::make_shared<ast::FieldAccessExpr>(sirInField->Name,
                                                        ast::Offsets{ast::cartesian, +2, 0, 0});

  rhsInT1->setID(target->nextUID());
  rhsInT2->setID(target->nextUID());
  rhsInT3->setID(target->nextUID());
  rhsInT4->setID(target->nextUID());

  auto lhsOut = std::make_shared<ast::FieldAccessExpr>(sirOutField->Name);
  lhsOut->setID(target->nextUID());

  auto rhsTmpT1 = std::make_shared<ast::FieldAccessExpr>(sirTmpField->Name,
                                                         ast::Offsets{ast::cartesian, 0, -1, 0});
  auto rhsTmpT2 = std::make_shared<ast::FieldAccessExpr>(sirTmpField->Name,
                                                         ast::Offsets{ast::cartesian, 0, +1, 0});
  auto rhsTmpT3 = std::make_shared<ast::FieldAccessExpr>(sirTmpField->Name,
                                                         ast::Offsets{ast::cartesian, -1, 0, 0});
  auto rhsTmpT4 = std::make_shared<ast::FieldAccessExpr>(sirTmpField->Name,
                                                         ast::Offsets{ast::cartesian, +1, 0, 0});

  rhsTmpT1->setID(target->nextUID());
  rhsTmpT2->setID(target->nextUID());
  rhsTmpT3->setID(target->nextUID());
  rhsTmpT4->setID(target->nextUID());

  int inFieldID = target->getMetaData().addField(iir::FieldAccessType::APIField, sirInField->Name,
                                                 std::move(sirInField->Dimensions));
  int tmpFieldID =
      target->getMetaData().addField(iir::FieldAccessType::StencilTemporary, sirTmpField->Name,
                                     std::move(sirTmpField->Dimensions));
  int outFieldID = target->getMetaData().addField(iir::FieldAccessType::APIField, sirOutField->Name,
                                                  std::move(sirOutField->Dimensions));

  lhsTmp->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(tmpFieldID);
  rhsInT1->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(inFieldID);
  rhsInT2->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(inFieldID);
  rhsInT3->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(inFieldID);
  rhsInT4->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(inFieldID);

  lhsOut->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(outFieldID);
  rhsTmpT1->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(tmpFieldID);
  rhsTmpT2->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(tmpFieldID);
  rhsTmpT3->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(tmpFieldID);
  rhsTmpT4->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(tmpFieldID);

  auto plusIn1 = std::make_shared<ast::BinaryOperator>(rhsInT1, std::string("+"), rhsInT2);
  auto plusIn2 = std::make_shared<ast::BinaryOperator>(rhsInT3, std::string("+"), rhsInT4);
  auto plusIn3 = std::make_shared<ast::BinaryOperator>(plusIn1, std::string("+"), plusIn2);

  plusIn1->setID(target->nextUID());
  plusIn2->setID(target->nextUID());
  plusIn3->setID(target->nextUID());

  auto assignmentTmpIn = std::make_shared<ast::AssignmentExpr>(lhsTmp, plusIn3);
  assignmentTmpIn->setID(target->nextUID());

  auto stmt1 = iir::makeExprStmt(assignmentTmpIn);
  stmt1->setID(target->nextUID());

  // Add the accesses:
  iir::Accesses callerAccesses1;
  callerAccesses1.addWriteExtent(tmpFieldID, iir::Extents(ast::cartesian, 0, 0, 0, 0, 0, 0));
  callerAccesses1.addReadExtent(inFieldID, iir::Extents(ast::cartesian, -2, 2, -2, 2, 0, 0));
  stmt1->getData<iir::IIRStmtData>().CallerAccesses =
      std::make_optional(std::move(callerAccesses1));

  // And add the statement to it
  IIRDoMethod1->getAST().push_back(std::move(stmt1));
  IIRDoMethod1->updateLevel();

  auto plusTmp1 = std::make_shared<ast::BinaryOperator>(rhsTmpT1, std::string("+"), rhsTmpT2);
  auto plusTmp2 = std::make_shared<ast::BinaryOperator>(rhsTmpT3, std::string("+"), rhsTmpT4);
  auto plusTmp3 = std::make_shared<ast::BinaryOperator>(plusTmp1, std::string("+"), plusTmp2);

  plusTmp1->setID(target->nextUID());
  plusTmp2->setID(target->nextUID());
  plusTmp3->setID(target->nextUID());

  auto assignmentOutTmp = std::make_shared<ast::AssignmentExpr>(lhsOut, plusTmp3);
  assignmentOutTmp->setID(target->nextUID());

  auto stmt2 = iir::makeExprStmt(assignmentOutTmp);
  stmt2->setID(target->nextUID());

  // Add the accesses to the statement:
  iir::Accesses callerAccesses2;
  callerAccesses2.addWriteExtent(outFieldID, iir::Extents(ast::cartesian, 0, 0, 0, 0, 0, 0));
  callerAccesses2.addReadExtent(tmpFieldID, iir::Extents(ast::cartesian, -1, 1, -1, 1, 0, 0));
  stmt2->getData<iir::IIRStmtData>().CallerAccesses =
      std::make_optional(std::move(callerAccesses2));
  // And add the statement to it
  IIRDoMethod2->getAST().push_back(std::move(stmt2));
  IIRDoMethod2->updateLevel();

  // Add the control flow descriptor to the IIR
  auto stencilCall = std::make_shared<ast::StencilCall>("generatedDriver");
  stencilCall->Args.push_back(sirInField->Name);
  // stencilCall->Args.push_back(sirTmpField->Name);
  stencilCall->Args.push_back(sirOutField->Name);
  auto placeholderStencil = std::make_shared<ast::StencilCall>(
      iir::InstantiationHelper::makeStencilCallCodeGenName(stencilID));
  auto stencilCallDeclStmt = iir::makeStencilCallDeclStmt(placeholderStencil);
  // Register the call and set it as a replacement for the next vertical region
  target->getMetaData().addStencilCallStmt(stencilCallDeclStmt, stencilID);
  target->getIIR()->getControlFlowDescriptor().insertStmt(stencilCallDeclStmt);

  ///////////////// Generation of the Metadata

  target->getMetaData().setStencilName("generated");

  for(const auto& MS : iterateIIROver<iir::MultiStage>(*(target->getIIR()))) {
    MS->update(iir::NodeUpdateType::levelAndTreeAbove);
  }
  // Iterate all statements (top -> bottom)
  for(const auto& stagePtr : iterateIIROver<iir::Stage>(*(target->getIIR()))) {
    iir::Stage& stage = *stagePtr;
    for(const auto& doMethod : stage.getChildren()) {
      doMethod->update(iir::NodeUpdateType::level);
    }
    stage.update(iir::NodeUpdateType::level);
  }
  for(const auto& MSPtr : iterateIIROver<iir::Stage>(*(target->getIIR()))) {
    MSPtr->update(iir::NodeUpdateType::levelAndTreeAbove);
  }

  return target;
}

std::shared_ptr<dawn::iir::StencilInstantiation> createUnstructuredSumEdgeToCellsIIRInMemory() {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto in_f = b.field("in_field", LocType::Edges);
  auto out_f = b.field("out_field", LocType::Cells);
  auto cnt = b.localvar("cnt", dawn::BuiltinTypeID::Integer);

  auto stencilInstantiation = b.build(
      "generated",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(LocType::Edges, b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                             b.stmt(b.assignExpr(b.at(in_f), b.lit(10))))),
          b.stage(
              LocType::Cells,
              b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                         b.stmt(b.assignExpr(
                             b.at(out_f), b.reduceOverNeighborExpr(
                                              Op::plus, b.at(in_f, HOffsetType::withOffset, 0),
                                              b.lit(0.), {LocType::Cells, LocType::Edges}))))))));
  return stencilInstantiation;
}

std::shared_ptr<dawn::iir::StencilInstantiation> createUnstructuredMixedCopies() {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto in_c = b.field("in_c", LocType::Cells);
  auto out_c = b.field("out_c", LocType::Cells);
  auto in_e = b.field("in_e", LocType::Edges);
  auto out_e = b.field("out_e", LocType::Edges);

  auto stencilInstantiation = b.build(
      "generated",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(LocType::Cells, b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                             b.stmt(b.assignExpr(b.at(out_c), b.at(in_c))))),
          b.stage(LocType::Edges, b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                             b.stmt(b.assignExpr(b.at(out_e), b.at(in_e))))))));
  return stencilInstantiation;
}
