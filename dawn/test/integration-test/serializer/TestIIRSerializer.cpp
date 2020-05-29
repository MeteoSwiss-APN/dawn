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

#include "dawn/AST/ASTExpr.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Support/STLExtras.h"
#include "dawn/Support/Type.h"
#include "dawn/Unittest/IIRBuilder.h"
#include "dawn/Unittest/UnittestUtils.h"
#include <gtest/gtest.h>
#include <memory>
#include <optional>

using namespace dawn;

namespace {
#define IIR_EARLY_EXIT(value)                                                                      \
  if(!value)                                                                                       \
    return false;

#define IIR_EXPECT_IMPL(iir1, iir2, VALUE)                                                         \
  do {                                                                                             \
    EXPECT_##VALUE(compareStencilInstantiations(iir1, iir2));                                      \
  } while(0);

#define IIR_EXPECT_EQ(iir1, iir2) IIR_EXPECT_IMPL((iir1), (iir2), TRUE)
#define IIR_EXPECT_NE(iir1, iir2) IIR_EXPECT_IMPL((iir1), (iir2), FALSE)

bool compareIIRs(iir::IIR* lhs, iir::IIR* rhs) {
  IIR_EARLY_EXIT((lhs->getGridType() == rhs->getGridType()));
  IIR_EARLY_EXIT(lhs->checkTreeConsistency());
  IIR_EARLY_EXIT(rhs->checkTreeConsistency());
  // checking the stencils
  for(int stencils = 0, size = lhs->getChildren().size(); stencils < size; ++stencils) {
    const auto& lhsStencil = lhs->getChild(stencils);
    const auto& rhsStencil = rhs->getChild(stencils);
    IIR_EARLY_EXIT((lhsStencil->getStencilAttributes() == rhsStencil->getStencilAttributes()));
    IIR_EARLY_EXIT((lhsStencil->getStencilID() == rhsStencil->getStencilID()));

    // checking each of the multistages
    for(int mssidx = 0, mssSize = lhsStencil->getChildren().size(); mssidx < mssSize; ++mssidx) {
      const auto& lhsMSS = lhsStencil->getChild(mssidx);
      const auto& rhsMSS = rhsStencil->getChild(mssidx);
      IIR_EARLY_EXIT((lhsMSS->getLoopOrder() == rhsMSS->getLoopOrder()));
      IIR_EARLY_EXIT((lhsMSS->getID() == rhsMSS->getID()));
      IIR_EARLY_EXIT((lhsMSS->getCaches().size() == rhsMSS->getCaches().size()));
      for(const auto& lhsPair : lhsMSS->getCaches()) {
        IIR_EARLY_EXIT(rhsMSS->getCaches().count(lhsPair.first));
        auto rhsValue = rhsMSS->getCaches().at(lhsPair.first);
        IIR_EARLY_EXIT((rhsValue == lhsPair.second));
      }
      // checking each of the stages
      for(int stageidx = 0, stageSize = lhsMSS->getChildren().size(); stageidx < stageSize;
          ++stageidx) {
        const auto& lhsStage = lhsMSS->getChild(stageidx);
        const auto& rhsStage = rhsMSS->getChild(stageidx);
        IIR_EARLY_EXIT((lhsStage->getStageID() == rhsStage->getStageID()));

        // checking each of the doMethods
        for(int doMethodidx = 0, doMethodSize = lhsStage->getChildren().size();
            doMethodidx < doMethodSize; ++doMethodidx) {
          const auto& lhsDoMethod = lhsStage->getChild(doMethodidx);
          const auto& rhsDoMethod = rhsStage->getChild(doMethodidx);
          IIR_EARLY_EXIT((lhsDoMethod->getID() == rhsDoMethod->getID()));
          IIR_EARLY_EXIT((lhsDoMethod->getInterval() == rhsDoMethod->getInterval()));

          // checking each of the statements
          for(int stmtidx = 0, stmtSize = lhsDoMethod->getAST().getStatements().size();
              stmtidx < stmtSize; ++stmtidx) {
            const auto& lhsStmt = lhsDoMethod->getAST().getStatements()[stmtidx];
            const auto& rhsStmt = rhsDoMethod->getAST().getStatements()[stmtidx];
            // check the statement (and its data)
            IIR_EARLY_EXIT((lhsStmt->equals(rhsStmt.get())));
          }
        }

        IIR_EARLY_EXIT((lhsStage->getLocationType() == rhsStage->getLocationType()));
      }
    }
  }
  const auto& lhsControlFlowStmts = lhs->getControlFlowDescriptor().getStatements();
  const auto& rhsControlFlowStmts = rhs->getControlFlowDescriptor().getStatements();

  IIR_EARLY_EXIT((lhsControlFlowStmts.size() == rhsControlFlowStmts.size()));
  for(int i = 0, size = lhsControlFlowStmts.size(); i < size; ++i) {
    // check the statement (and its data)
    if(!lhsControlFlowStmts[i]->equals(rhsControlFlowStmts[i].get()))
      return false;
  }

  return true;
}

bool compareMetaData(iir::StencilMetaInformation& lhs, iir::StencilMetaInformation& rhs) {
  IIR_EARLY_EXIT((lhs.getAccessesOfType<iir::FieldAccessType::Literal>() ==
                  rhs.getAccessesOfType<iir::FieldAccessType::Literal>()));
  IIR_EARLY_EXIT((lhs.getAccessesOfType<iir::FieldAccessType::Field>() ==
                  rhs.getAccessesOfType<iir::FieldAccessType::Field>()));
  IIR_EARLY_EXIT((lhs.getAccessesOfType<iir::FieldAccessType::APIField>() ==
                  rhs.getAccessesOfType<iir::FieldAccessType::APIField>()));
  IIR_EARLY_EXIT((lhs.getAccessesOfType<iir::FieldAccessType::StencilTemporary>() ==
                  rhs.getAccessesOfType<iir::FieldAccessType::StencilTemporary>()));
  IIR_EARLY_EXIT((lhs.getAccessesOfType<iir::FieldAccessType::GlobalVariable>() ==
                  rhs.getAccessesOfType<iir::FieldAccessType::GlobalVariable>()));

  // we compare the content of the maps since the shared-ptr's are not the same
  IIR_EARLY_EXIT((lhs.getFieldNameToBCMap().size() == rhs.getFieldNameToBCMap().size()));
  for(const auto& lhsPair : lhs.getFieldNameToBCMap()) {
    IIR_EARLY_EXIT(rhs.getFieldNameToBCMap().count(lhsPair.first));
    auto rhsValue = rhs.getFieldNameToBCMap().at(lhsPair.first);
    IIR_EARLY_EXIT(rhsValue->equals(lhsPair.second.get()));
  }
  IIR_EARLY_EXIT((lhs.getFieldIDToDimsMap() == rhs.getFieldIDToDimsMap()));
  IIR_EARLY_EXIT((lhs.getStencilLocation() == rhs.getStencilLocation()));
  IIR_EARLY_EXIT((lhs.getStencilName() == rhs.getStencilName()));
  IIR_EARLY_EXIT((lhs.getFileName() == rhs.getFileName()));

  // we compare the content of the maps since the shared-ptr's are not the same
  IIR_EARLY_EXIT((lhs.getStencilIDToStencilCallMap().getDirectMap().size() ==
                  rhs.getStencilIDToStencilCallMap().getDirectMap().size()));
  for(const auto& lhsPair : lhs.getStencilIDToStencilCallMap().getDirectMap()) {
    IIR_EARLY_EXIT(rhs.getStencilIDToStencilCallMap().getDirectMap().count(lhsPair.first));
    auto rhsValue = rhs.getStencilIDToStencilCallMap().getDirectMap().at(lhsPair.first);
    IIR_EARLY_EXIT(rhsValue->equals(lhsPair.second.get()));
  }
  return true;
}

bool compareStencilInstantiations(const std::shared_ptr<iir::StencilInstantiation>& lhs,
                                  const std::shared_ptr<iir::StencilInstantiation>& rhs) {
  IIR_EARLY_EXIT(compareIIRs(lhs->getIIR().get(), rhs->getIIR().get()));
  IIR_EARLY_EXIT(compareMetaData(lhs->getMetaData(), rhs->getMetaData()));
  return true;
}

class createEmptyOptimizerContext : public ::testing::Test {
protected:
  virtual void SetUp() override {
    std::shared_ptr<SIR> sir = std::make_shared<SIR>(ast::GridType::Cartesian);
    dawn::OptimizerContext::OptimizerContextOptions options;
    context_ = std::make_unique<OptimizerContext>(options, sir);
  }
  virtual void TearDown() override {}
  std::unique_ptr<OptimizerContext> context_;
};

class IIRSerializerTest : public createEmptyOptimizerContext {
protected:
  virtual void SetUp() override {
    createEmptyOptimizerContext::SetUp();
    referenceInstantiation = std::make_shared<iir::StencilInstantiation>(
        context_->getSIR()->GridType, context_->getSIR()->GlobalVariableMap,
        context_->getSIR()->StencilFunctions);
  }
  virtual void TearDown() override { referenceInstantiation.reset(); }

  std::shared_ptr<iir::StencilInstantiation> serializeAndDeserializeRef() {
    return IIRSerializer::deserializeFromString(
        IIRSerializer::serializeToString(referenceInstantiation));
  }

  std::shared_ptr<iir::StencilInstantiation> referenceInstantiation;
};

TEST_F(IIRSerializerTest, EmptySetup) {
  auto desired = serializeAndDeserializeRef();
  IIR_EXPECT_EQ(desired, referenceInstantiation);
  desired->getMetaData().insertAccessOfType(iir::FieldAccessType::InterStencilTemporary, 10,
                                            "name");
  IIR_EXPECT_NE(desired, referenceInstantiation);
}
TEST_F(IIRSerializerTest, SimpleDataStructures) {
  //===------------------------------------------------------------------------------------------===
  // Checking inserts into the various maps
  //===------------------------------------------------------------------------------------------===
  referenceInstantiation->getMetaData().addAccessIDNamePair(1, "test");
  IIR_EXPECT_EQ(serializeAndDeserializeRef(), referenceInstantiation);

  referenceInstantiation->getMetaData().insertAccessOfType(iir::FieldAccessType::Literal, -5,
                                                           "test");
  IIR_EXPECT_EQ(serializeAndDeserializeRef(), referenceInstantiation);

  referenceInstantiation->getMetaData().insertAccessOfType(iir::FieldAccessType::Field, 712,
                                                           "field0");
  IIR_EXPECT_EQ(serializeAndDeserializeRef(), referenceInstantiation);

  referenceInstantiation->getMetaData().insertAccessOfType(iir::FieldAccessType::APIField, 10,
                                                           "field1");
  referenceInstantiation->getMetaData().insertAccessOfType(iir::FieldAccessType::APIField, 12,
                                                           "field2");
  auto deserializedStencilInstantiaion = serializeAndDeserializeRef();
  IIR_EXPECT_EQ(deserializedStencilInstantiaion, referenceInstantiation);

  // check that ordering is preserved
  referenceInstantiation->getMetaData().removeAccessID(12);
  referenceInstantiation->getMetaData().removeAccessID(10);

  referenceInstantiation->getMetaData().insertAccessOfType(iir::FieldAccessType::APIField, 12,
                                                           "field1");
  referenceInstantiation->getMetaData().insertAccessOfType(iir::FieldAccessType::APIField, 10,
                                                           "field2");

  IIR_EXPECT_NE(deserializedStencilInstantiaion, referenceInstantiation);

  referenceInstantiation->getMetaData().insertAccessOfType(iir::FieldAccessType::StencilTemporary,
                                                           713, "field4");
  IIR_EXPECT_EQ(serializeAndDeserializeRef(), referenceInstantiation);

  referenceInstantiation->getMetaData().addFieldVersionIDPair(5, 7);
  referenceInstantiation->getMetaData().addFieldVersionIDPair(5, 8);
  IIR_EXPECT_EQ(serializeAndDeserializeRef(), referenceInstantiation);

  referenceInstantiation->getMetaData().setFileName("fileName");
  IIR_EXPECT_EQ(serializeAndDeserializeRef(), referenceInstantiation);
  referenceInstantiation->getMetaData().setStencilName("stencilName");
  IIR_EXPECT_EQ(serializeAndDeserializeRef(), referenceInstantiation);
  referenceInstantiation->getMetaData().setStencilLocation(SourceLocation{1, 2});
  IIR_EXPECT_EQ(serializeAndDeserializeRef(), referenceInstantiation);
}

TEST_F(IIRSerializerTest, ComplexStrucutes) {
  auto scStmt = iir::makeStencilCallDeclStmt(std::make_shared<ast::StencilCall>("me"));
  scStmt->getSourceLocation().Line = 10;
  scStmt->getSourceLocation().Column = 12;
  referenceInstantiation->getIIR()->getControlFlowDescriptor().insertStmt(scStmt);
  IIR_EXPECT_EQ(serializeAndDeserializeRef(), referenceInstantiation);

  auto stmt = iir::makeStencilCallDeclStmt(std::make_shared<ast::StencilCall>("test"));
  IIR_EXPECT_EQ(serializeAndDeserializeRef(), referenceInstantiation);

  auto bcstmt = iir::makeBoundaryConditionDeclStmt("callee");
  bcstmt->getFields().push_back("field1");
  bcstmt->getFields().push_back("field2");
  referenceInstantiation->getMetaData().addFieldBC("bc", bcstmt);
  IIR_EXPECT_EQ(serializeAndDeserializeRef(), referenceInstantiation);
}

TEST_F(IIRSerializerTest, IIRTestsStageLocationType) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto in_c = b.field("in_c", LocType::Cells);
  auto out_c = b.field("out_c", LocType::Cells);
  auto in_v = b.field("in_v", LocType::Vertices);
  auto out_v = b.field("out_v", LocType::Vertices);

  std::string stencilName("testSerializationStageLocationType");

  auto stencil_instantiation = b.build(
      stencilName.c_str(),
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(LocType::Cells, b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                             b.stmt(b.assignExpr(b.at(out_c), b.at(in_c))))),
          b.stage(LocType::Vertices,
                  b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.stmt(b.assignExpr(b.at(out_v), b.at(in_v))))))));

  auto deserializedAndSerialized =
      IIRSerializer::deserializeFromString(IIRSerializer::serializeToString(stencil_instantiation));

  IIR_EXPECT_EQ(stencil_instantiation, deserializedAndSerialized);
}

TEST_F(IIRSerializerTest, IIRTestsReduce) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto in_f = b.field("in_field", LocType::Edges);
  auto out_f = b.field("out_field", LocType::Cells);

  std::string stencilName("testSerializationReduce");

  auto stencil_instantiation =
      b.build(stencilName.c_str(),
              b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(
                      dawn::sir::Interval::Start, dawn::sir::Interval::End,
                      b.stmt(b.assignExpr(b.at(out_f),
                                          b.reduceOverNeighborExpr(
                                              Op::plus, b.at(in_f, HOffsetType::withOffset, 0),
                                              b.lit(0.), {LocType::Cells, LocType::Edges}))))))));

  auto deserializedAndSerialized =
      IIRSerializer::deserializeFromString(IIRSerializer::serializeToString(stencil_instantiation));

  IIR_EXPECT_EQ(stencil_instantiation, deserializedAndSerialized);
}

TEST_F(IIRSerializerTest, IIRTestsWeightedReduce) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto in_f = b.field("in_field", LocType::Edges);
  auto out_f = b.field("out_field", LocType::Cells);

  std::string stencilName("testSerializationReduceWeights");

  auto stencil_instantiation = b.build(
      stencilName.c_str(),
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(b.doMethod(
              dawn::sir::Interval::Start, dawn::sir::Interval::End,
              b.stmt(b.assignExpr(b.at(out_f), b.reduceOverNeighborExpr(
                                                   Op::plus, b.at(in_f, HOffsetType::withOffset, 0),
                                                   b.lit(0.), {LocType::Cells, LocType::Edges},
                                                   std::vector<float>({1., 2., 3., 4.})))),
              b.stmt(b.assignExpr(b.at(out_f), b.reduceOverNeighborExpr(
                                                   Op::plus, b.at(in_f, HOffsetType::withOffset, 0),
                                                   b.lit(0.), {LocType::Cells, LocType::Edges},
                                                   std::vector<double>({1., 2., 3., 4.})))),
              b.stmt(b.assignExpr(b.at(out_f), b.reduceOverNeighborExpr(
                                                   Op::plus, b.at(in_f, HOffsetType::withOffset, 0),
                                                   b.lit(0.), {LocType::Cells, LocType::Edges},
                                                   std::vector<int>({1, 2, 3, 4})))))))));

  auto deserializedAndSerialized =
      IIRSerializer::deserializeFromString(IIRSerializer::serializeToString(stencil_instantiation));

  IIR_EXPECT_EQ(stencil_instantiation, deserializedAndSerialized);
}

TEST_F(IIRSerializerTest, IIRTestsGeneralWeightedReduce) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto in_f = b.field("in_field", LocType::Edges);
  auto out_f = b.field("out_field", LocType::Cells);
  auto aux0_f = b.field("aux0_field", LocType::Cells);
  auto aux1_f = b.field("aux1_field", LocType::Cells);

  std::string stencilName("testSerializationReduceWeights");

  auto stencil_instantiation = b.build(
      stencilName.c_str(),
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(b.doMethod(
              dawn::sir::Interval::Start, dawn::sir::Interval::End,
              b.stmt(b.assignExpr(b.at(out_f),
                                  b.reduceOverNeighborExpr(
                                      Op::plus, b.at(in_f, HOffsetType::withOffset, 0), b.lit(0.),
                                      {LocType::Cells, LocType::Edges},
                                      {b.at(aux0_f), b.at(aux0_f), b.at(aux1_f), b.at(aux1_f)}))),
              b.stmt(b.assignExpr(
                  b.at(out_f), b.reduceOverNeighborExpr(
                                   Op::plus, b.at(in_f, HOffsetType::withOffset, 0), b.lit(0.),
                                   {LocType::Cells, LocType::Edges},
                                   {b.binaryExpr(b.at(aux0_f), b.at(aux0_f), Op::multiply),
                                    b.binaryExpr(b.at(aux0_f), b.at(aux1_f), Op::multiply),
                                    b.binaryExpr(b.at(aux1_f), b.at(aux0_f), Op::multiply),
                                    b.binaryExpr(b.at(aux1_f), b.at(aux1_f), Op::multiply)}))))))));

  auto deserializedAndSerialized =
      IIRSerializer::deserializeFromString(IIRSerializer::serializeToString(stencil_instantiation));

  IIR_EXPECT_EQ(stencil_instantiation, deserializedAndSerialized);
}

TEST_F(IIRSerializerTest, IIRTests) {
  sir::Attr attributes;
  attributes.set(sir::Attr::Kind::MergeStages);
  referenceInstantiation->getIIR()->insertChild(
      std::make_unique<iir::Stencil>(referenceInstantiation->getMetaData(), attributes, 10),
      referenceInstantiation->getIIR());
  const auto& IIRStencil = referenceInstantiation->getIIR()->getChild(0);
  auto deserialized = serializeAndDeserializeRef();
  IIR_EXPECT_EQ(deserialized, referenceInstantiation);
  IIRStencil->getStencilAttributes().set(sir::Attr::Kind::NoCodeGen);
  IIR_EXPECT_NE(deserialized, referenceInstantiation);

  (IIRStencil)
      ->insertChild(std::make_unique<iir::MultiStage>(referenceInstantiation->getMetaData(),
                                                      iir::LoopOrderKind::Backward));
  const auto& IIRMSS = (IIRStencil)->getChild(0);
  IIRMSS->getCaches().emplace(10, iir::Cache(iir::Cache::CacheType::IJ, iir::Cache::IOPolicy::fill,
                                             10, std::nullopt, std::nullopt, std::nullopt));
  deserialized = serializeAndDeserializeRef();
  IIR_EXPECT_EQ(deserialized, referenceInstantiation);
  IIRMSS->setLoopOrder(iir::LoopOrderKind::Forward);
  IIR_EXPECT_NE(deserialized, referenceInstantiation);

  IIRMSS->insertChild(std::make_unique<iir::Stage>(referenceInstantiation->getMetaData(), 12));
  const auto& IIRStage = IIRMSS->getChild(0);
  IIR_EXPECT_EQ(serializeAndDeserializeRef(), referenceInstantiation);

  (IIRStage)->insertChild(std::make_unique<iir::DoMethod>(iir::Interval(1, 5, 0, 1),
                                                          referenceInstantiation->getMetaData()));
  IIR_EXPECT_EQ(serializeAndDeserializeRef(), referenceInstantiation);

  auto& IIRDoMethod = (IIRStage)->getChild(0);
  auto expr = std::make_shared<iir::VarAccessExpr>("name");
  expr->getData<iir::IIRAccessExprData>().AccessID = std::make_optional<int>(42);
  auto stmt = iir::makeExprStmt(expr);
  stmt->setID(22);
  iir::Accesses stmtAccesses;
  iir::Extents extents(ast::Offsets{ast::cartesian});
  stmtAccesses.addReadExtent(42, extents);
  stmt->getData<iir::IIRStmtData>().CallerAccesses = std::make_optional(std::move(stmtAccesses));

  IIRDoMethod->getAST().push_back(std::move(stmt));
  std::string varName = "foo";
  auto varDeclStmt = iir::makeVarDeclStmt(dawn::Type(BuiltinTypeID::Float), varName, 0, "=",
                                          std::vector<std::shared_ptr<iir::Expr>>{expr->clone()});
  iir::Accesses varDeclStmtAccesses;
  varDeclStmtAccesses.addWriteExtent(33, extents);
  varDeclStmt->getData<iir::IIRStmtData>().CallerAccesses =
      std::make_optional(std::move(varDeclStmtAccesses));
  varDeclStmt->getData<iir::VarDeclStmtData>().AccessID = std::make_optional<int>(33);

  IIRDoMethod->getAST().push_back(std::move(varDeclStmt));

  deserialized = serializeAndDeserializeRef();
  IIR_EXPECT_EQ(deserialized, referenceInstantiation);
  auto deserializedExprStmt =
      std::dynamic_pointer_cast<iir::ExprStmt>(getNthStmt(getFirstDoMethod(deserialized), 0));
  deserializedExprStmt->getData<iir::IIRStmtData>().CallerAccesses->addReadExtent(50, extents);
  IIR_EXPECT_NE(deserialized, referenceInstantiation);
  deserialized = serializeAndDeserializeRef();
  auto deserializedVarAccessExpr = std::dynamic_pointer_cast<iir::VarAccessExpr>(
      std::dynamic_pointer_cast<iir::ExprStmt>(getNthStmt(getFirstDoMethod(deserialized), 0))
          ->getExpr());
  deserializedVarAccessExpr->getData<iir::IIRAccessExprData>().AccessID =
      std::make_optional<int>(50);
  IIR_EXPECT_NE(deserialized, referenceInstantiation);
  deserialized = serializeAndDeserializeRef();
  auto deserializedVarDeclStmt =
      std::dynamic_pointer_cast<iir::VarDeclStmt>(getNthStmt(getFirstDoMethod(deserialized), 1));
  deserializedVarDeclStmt->getData<iir::VarDeclStmtData>().AccessID = std::make_optional<int>(34);
  IIR_EXPECT_NE(deserialized, referenceInstantiation);
}

TEST_F(IIRSerializerTest, IterationSpace) {
  using namespace dawn::iir;

  CartesianIIRBuilder b;
  auto in_f = b.field("in_f", FieldType::ijk);
  auto out_f = b.field("out_f", FieldType::ijk);

  auto instantiation =
      b.build("iteration_space",
              b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                     b.block(b.stmt(b.assignExpr(b.at(out_f), b.at(in_f)))))),
                  b.stage(1, {0, 2},
                          b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                     b.block(b.stmt(b.assignExpr(b.at(out_f), b.lit(10)))))))));

  std::string serializedIIR = IIRSerializer::serializeToString(instantiation);
  auto deserialized = IIRSerializer::deserializeFromString(serializedIIR);
  std::string deserializedIIR = IIRSerializer::serializeToString(deserialized);

  IIR_EXPECT_EQ(instantiation, deserialized);
}

} // anonymous namespace
