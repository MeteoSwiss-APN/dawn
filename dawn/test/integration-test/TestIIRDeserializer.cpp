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

#include "dawn/AST/GridType.h"
#include "dawn/Compiler/DawnCompiler.h"
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
#include "dawn/Optimizer/PassComputeStageExtents.h"
#include "dawn/Optimizer/PassSetStageName.h"
#include "dawn/Optimizer/PassTemporaryType.h"
#include "dawn/Optimizer/StatementMapper.h"
#include "dawn/SIR/ASTFwd.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/STLExtras.h"

#include <gtest/gtest.h>
#include <memory>
#include <stack>
#include <string>
#include <unistd.h>

#include "GenerateInMemoryStencils.h"

using namespace dawn;

namespace {

void compareIIRstructures(iir::IIR* lhs, iir::IIR* rhs) {
  EXPECT_TRUE(lhs->checkTreeConsistency());
  EXPECT_TRUE(rhs->checkTreeConsistency());
  // checking the stencils
  ASSERT_EQ(lhs->getChildren().size(), rhs->getChildren().size());
  for(int stencils = 0, size = lhs->getChildren().size(); stencils < size; ++stencils) {
    const auto& lhsStencil = lhs->getChild(stencils);
    const auto& rhsStencil = rhs->getChild(stencils);
    EXPECT_EQ(lhsStencil->getStencilAttributes(), rhsStencil->getStencilAttributes());
    EXPECT_EQ(lhsStencil->getStencilID(), rhsStencil->getStencilID());

    // checking each of the multistages
    ASSERT_EQ(lhsStencil->getChildren().size(), rhsStencil->getChildren().size());
    for(int mssidx = 0, mssSize = lhsStencil->getChildren().size(); mssidx < mssSize; ++mssidx) {
      const auto& lhsMSS = lhsStencil->getChild(mssidx);
      const auto& rhsMSS = rhsStencil->getChild(mssidx);
      EXPECT_EQ(lhsMSS->getLoopOrder(), rhsMSS->getLoopOrder());
      EXPECT_EQ(lhsMSS->getID(), rhsMSS->getID());

      // checking each of the stages
      ASSERT_EQ(lhsMSS->getChildren().size(), rhsMSS->getChildren().size());
      for(int stageidx = 0, stageSize = lhsMSS->getChildren().size(); stageidx < stageSize;
          ++stageidx) {
        const auto& lhsStage = lhsMSS->getChild(stageidx);
        const auto& rhsStage = rhsMSS->getChild(stageidx);
        EXPECT_EQ(lhsStage->getStageID(), rhsStage->getStageID());

        // checking each of the doMethods
        ASSERT_EQ(lhsStage->getChildren().size(), rhsStage->getChildren().size());
        for(int doMethodIdx = 0, doMethodSize = lhsStage->getChildren().size();
            doMethodIdx < doMethodSize; ++doMethodIdx) {
          const auto& lhsDoMethod = lhsStage->getChild(doMethodIdx);
          const auto& rhsDoMethod = rhsStage->getChild(doMethodIdx);
          EXPECT_EQ(lhsDoMethod->getID(), rhsDoMethod->getID());
          EXPECT_EQ(lhsDoMethod->getInterval(), rhsDoMethod->getInterval());

          // checking each of the statements
          ASSERT_EQ(lhsDoMethod->getAST().getStatements().size(),
                    rhsDoMethod->getAST().getStatements().size());
          for(int stmtidx = 0, stmtSize = lhsDoMethod->getAST().getStatements().size();
              stmtidx < stmtSize; ++stmtidx) {
            const auto& lhsStmt = lhsDoMethod->getAST().getStatements()[stmtidx];
            const auto& rhsStmt = rhsDoMethod->getAST().getStatements()[stmtidx];
            // check the statement (and its data)
            EXPECT_TRUE(lhsStmt->equals(rhsStmt.get()));
          }
        }
      }
    }
  }
  const auto& lhsControlFlowStmts = lhs->getControlFlowDescriptor().getStatements();
  const auto& rhsControlFlowStmts = rhs->getControlFlowDescriptor().getStatements();

  ASSERT_EQ(lhsControlFlowStmts.size(), rhsControlFlowStmts.size());
  for(int i = 0, size = lhsControlFlowStmts.size(); i < size; ++i) {
    EXPECT_TRUE(lhsControlFlowStmts[i]->equals(rhsControlFlowStmts[i].get()));
  }
}

void compareMetaData(iir::StencilMetaInformation& lhs, iir::StencilMetaInformation& rhs) {
  EXPECT_EQ(lhs.getAccessesOfType<iir::FieldAccessType::Literal>(),
            rhs.getAccessesOfType<iir::FieldAccessType::Literal>());
  EXPECT_EQ(lhs.getAccessesOfType<iir::FieldAccessType::Field>(),
            rhs.getAccessesOfType<iir::FieldAccessType::Field>());
  EXPECT_EQ(lhs.getAccessesOfType<iir::FieldAccessType::APIField>(),
            rhs.getAccessesOfType<iir::FieldAccessType::APIField>());
  EXPECT_EQ(lhs.getAccessesOfType<iir::FieldAccessType::StencilTemporary>(),
            rhs.getAccessesOfType<iir::FieldAccessType::StencilTemporary>());
  EXPECT_EQ(lhs.getAccessesOfType<iir::FieldAccessType::GlobalVariable>(),
            rhs.getAccessesOfType<iir::FieldAccessType::GlobalVariable>());

  // we compare the content of the maps since the shared-ptr's are not the same
  ASSERT_EQ(lhs.getFieldNameToBCMap().size(), rhs.getFieldNameToBCMap().size());
  for(const auto& lhsPair : lhs.getFieldNameToBCMap()) {
    EXPECT_TRUE(rhs.getFieldNameToBCMap().count(lhsPair.first));
    auto rhsValue = rhs.getFieldNameToBCMap().at(lhsPair.first);
    EXPECT_TRUE(rhsValue->equals(lhsPair.second.get()));
  }
  EXPECT_EQ(lhs.getFieldIDToDimsMap(), rhs.getFieldIDToDimsMap());
  EXPECT_EQ(lhs.getStencilLocation(), rhs.getStencilLocation());
  EXPECT_EQ(lhs.getStencilName(), rhs.getStencilName());
  // file name makes little sense for in memory stencil
  // ASSERT_EQ(lhs.getFileName(), rhs.getFileName()));

  // we compare the content of the maps since the shared-ptr's are not the same
  ASSERT_EQ(lhs.getStencilIDToStencilCallMap().getDirectMap().size(),
            rhs.getStencilIDToStencilCallMap().getDirectMap().size());
  for(const auto& lhsPair : lhs.getStencilIDToStencilCallMap().getDirectMap()) {
    EXPECT_TRUE(rhs.getStencilIDToStencilCallMap().getDirectMap().count(lhsPair.first));
    auto rhsValue = rhs.getStencilIDToStencilCallMap().getDirectMap().at(lhsPair.first);
    EXPECT_TRUE(rhsValue->equals(lhsPair.second.get()));
  }
}

void compareDerivedInformation(iir::IIR* lhs, iir::IIR* rhs) {
  for(int stencils = 0, size = lhs->getChildren().size(); stencils < size; ++stencils) {
    const auto& lhsStencil = lhs->getChild(stencils);
    const auto& rhsStencil = rhs->getChild(stencils);

    EXPECT_EQ(lhsStencil->getStageDependencyGraph().get(),
              rhsStencil->getStageDependencyGraph().get());
    EXPECT_EQ(lhsStencil->getFields(), rhsStencil->getFields());

    ASSERT_EQ(lhsStencil->getChildren().size(), rhsStencil->getChildren().size());

    // checking each of the multistages
    for(int mssidx = 0, mssSize = lhsStencil->getChildren().size(); mssidx < mssSize; ++mssidx) {
      const auto& lhsMSS = lhsStencil->getChild(mssidx);
      const auto& rhsMSS = rhsStencil->getChild(mssidx);

      EXPECT_EQ(lhsMSS->getCaches(), rhsMSS->getCaches());
      EXPECT_EQ(lhsMSS->getFields(), rhsMSS->getFields());

      ASSERT_EQ(lhsMSS->getChildren().size(), rhsMSS->getChildren().size());

      // checking each of the stages
      for(int stageidx = 0, stageSize = lhsMSS->getChildren().size(); stageidx < stageSize;
          ++stageidx) {
        const auto& lhsStage = lhsMSS->getChild(stageidx);
        const auto& rhsStage = rhsMSS->getChild(stageidx);

        EXPECT_EQ(lhsStage->getFields(), rhsStage->getFields());
        EXPECT_EQ(lhsStage->getAllGlobalVariables(), rhsStage->getAllGlobalVariables());
        EXPECT_EQ(lhsStage->getGlobalVariables(), rhsStage->getGlobalVariables());
        EXPECT_EQ(lhsStage->getGlobalVariablesFromStencilFunctionCalls(),
                  rhsStage->getGlobalVariablesFromStencilFunctionCalls());
        EXPECT_EQ(lhsStage->getExtents(), rhsStage->getExtents());
        EXPECT_EQ(lhsStage->getRequiresSync(), rhsStage->getRequiresSync());

        ASSERT_EQ(lhsStage->getChildren().size(), rhsStage->getChildren().size());

        // checking each of the doMethods
        for(int doMethodIdx = 0, doMethodSize = lhsStage->getChildren().size();
            doMethodIdx < doMethodSize; ++doMethodIdx) {
          const auto& lhsDoMethod = lhsStage->getChild(doMethodIdx);
          const auto& rhsDoMethod = rhsStage->getChild(doMethodIdx);

          ASSERT_EQ(lhsDoMethod->getFields(), rhsDoMethod->getFields());
          ASSERT_EQ(lhsDoMethod->getDependencyGraph(), rhsDoMethod->getDependencyGraph());
        }
      }
    }
  }
}

std::shared_ptr<iir::StencilInstantiation> readIIRFromFile(OptimizerContext& optimizer,
                                                           const std::string& fname) {
  auto target = IIRSerializer::deserialize(fname, &optimizer, IIRSerializer::Format::Json);

  // this is whats actually to be tested.
  optimizer.restoreIIR("<restored>", target);
  return target;
}

void compareIIRs(std::shared_ptr<iir::StencilInstantiation> lhs,
                 std::shared_ptr<iir::StencilInstantiation> rhs) {
  // first compare the (structure of the) iirs, this is a precondition before we can actually check
  // the metadata / derived info
  compareIIRstructures(lhs->getIIR().get(), rhs->getIIR().get());

  // then we compare the meta data
  compareMetaData(lhs->getMetaData(), rhs->getMetaData());

  // and finally the derived info
  compareDerivedInformation(lhs->getIIR().get(), rhs->getIIR().get());
}

TEST(IIRDeserializerTest, CopyStencil) {
  OptimizerContext::OptimizerContextOptions optimizerOptions;
  DawnCompiler compiler;
  OptimizerContext optimizer(compiler.getDiagnostics(), optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Cartesian));

  // read IIR from file
  auto copy_stencil_from_file = readIIRFromFile(optimizer, "reference_iir/copy_stencil.iir");

  // generate IIR in memory
  UIDGenerator::getInstance()->reset();
  auto copy_stencil_memory = createCopyStencilIIRInMemory(optimizer);

  compareIIRs(copy_stencil_from_file, copy_stencil_memory);
}

TEST(IIRDeserializerTest, LapStencil) {
  OptimizerContext::OptimizerContextOptions optimizerOptions;
  DawnCompiler compiler;
  OptimizerContext optimizer(compiler.getDiagnostics(), optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Cartesian));

  // read IIR from file
  auto lap_stencil_from_file = readIIRFromFile(optimizer, "reference_iir/lap_stencil.iir");

  // generate IIR in memory
  UIDGenerator::getInstance()->reset();
  auto lap_stencil_memory = createLapStencilIIRInMemory(optimizer);

  compareIIRs(lap_stencil_from_file, lap_stencil_memory);
}

TEST(IIRDeserializerTest, UnstructuredSumEdgeToCells) {
  OptimizerContext::OptimizerContextOptions optimizerOptions;
  DawnCompiler compiler;
  OptimizerContext optimizer(compiler.getDiagnostics(), optimizerOptions,
                             std::make_shared<dawn::SIR>(dawn::ast::GridType::Unstructured));
  // read IIR from file
  auto from_file = readIIRFromFile(optimizer, "reference_iir/unstructured_sum_edge_to_cells.iir");

  // generate IIR in memory
  UIDGenerator::getInstance()->reset();
  auto in_memory = createUnstructuredSumEdgeToCellsIIRInMemory(optimizer);

  compareIIRs(from_file, in_memory);
}

} // anonymous namespace
