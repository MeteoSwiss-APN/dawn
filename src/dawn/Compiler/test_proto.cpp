#include "test_proto.h"

#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/ASTUtil.h"
#include "dawn/IIR/FieldAccessMetadata.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/InstantiationHelper.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Optimizer/AccessComputation.h"
#include "dawn/Optimizer/PassComputeStageExtents.h"
#include "dawn/Optimizer/PassTemporaryType.h"
#include "dawn/Optimizer/PassSetStageName.h"
#include "dawn/Optimizer/StatementMapper.h"
#include "dawn/SIR/ASTFwd.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/STLExtras.h"
#include <stack>

#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include <memory>
#include <string>

using namespace dawn;

bool compareIIRs(iir::IIR* lhs, iir::IIR* rhs) {
  assert(lhs->checkTreeConsistency());
  assert(rhs->checkTreeConsistency());
  // checking the stencils
  for(int stencils = 0, size = lhs->getChildren().size(); stencils < size; ++stencils) {
    const auto& lhsStencil = lhs->getChild(stencils);
    const auto& rhsStencil = rhs->getChild(stencils);
    assert((lhsStencil->getStencilAttributes() == rhsStencil->getStencilAttributes()));
    printf("%d %d\n", lhsStencil->getStencilID(), rhsStencil->getStencilID());
    assert((lhsStencil->getStencilID() == rhsStencil->getStencilID()));

    // checking each of the multistages
    for(int mssidx = 0, mssSize = lhsStencil->getChildren().size(); mssidx < mssSize; ++mssidx) {
      const auto& lhsMSS = lhsStencil->getChild(mssidx);
      const auto& rhsMSS = rhsStencil->getChild(mssidx);
      assert((lhsMSS->getLoopOrder() == rhsMSS->getLoopOrder()));
      assert((lhsMSS->getID() == rhsMSS->getID()));
      assert((lhsMSS->getCaches().size() == rhsMSS->getCaches().size()));
      for(const auto& lhsPair : lhsMSS->getCaches()) {
        assert(rhsMSS->getCaches().count(lhsPair.first));
        auto rhsValue = rhsMSS->getCaches().at(lhsPair.first);
        assert((rhsValue == lhsPair.second));
      }
      // checking each of the stages
      for(int stageidx = 0, stageSize = lhsMSS->getChildren().size(); stageidx < stageSize;
          ++stageidx) {
        const auto& lhsStage = lhsMSS->getChild(stageidx);
        const auto& rhsStage = rhsMSS->getChild(stageidx);
        assert((lhsStage->getStageID() == rhsStage->getStageID()));

        // checking each of the doMethods
        for(int doMethodidx = 0, doMethodSize = lhsStage->getChildren().size();
            doMethodidx < doMethodSize; ++doMethodidx) {
          const auto& lhsDoMethod = lhsStage->getChild(doMethodidx);
          const auto& rhsDoMethod = rhsStage->getChild(doMethodidx);
          assert((lhsDoMethod->getID() == rhsDoMethod->getID()));
          assert((lhsDoMethod->getInterval() == rhsDoMethod->getInterval()));

          // checking each of the StmtAccesspairs
          for(int stmtidx = 0, stmtSize = lhsDoMethod->getChildren().size(); stmtidx < stmtSize;
              ++stageidx) {
            const auto& lhsStmt = lhsDoMethod->getChild(stmtidx);
            const auto& rhsStmt = rhsDoMethod->getChild(stmtidx);
            // check the statement
            assert(
                (lhsStmt->getStatement()->ASTStmt->equals(rhsStmt->getStatement()->ASTStmt.get())));

            // check the accesses
            assert((lhsStmt->getCallerAccesses() == rhsStmt->getCallerAccesses()));
            if(lhsStmt->getCallerAccesses()) {
              for(const auto& lhsPair : rhsStmt->getCallerAccesses()->getReadAccesses()) {
                assert(
                    rhsStmt->getCallerAccesses()->getReadAccesses().count(lhsPair.first));
                auto rhsValue = rhsStmt->getCallerAccesses()->getReadAccesses().at(lhsPair.first);
                assert((rhsValue == lhsPair.second));
              }
              for(const auto& lhsPair : rhsStmt->getCallerAccesses()->getWriteAccesses()) {
                assert(
                    rhsStmt->getCallerAccesses()->getWriteAccesses().count(lhsPair.first));
                auto rhsValue = rhsStmt->getCallerAccesses()->getWriteAccesses().at(lhsPair.first);
                assert((rhsValue == lhsPair.second));
              }
            }
          }
        }
      }
    }
  }
  const auto& lhsControlFlowStmts = lhs->getControlFlowDescriptor().getStatements();
  const auto& rhsControlFlowStmts = rhs->getControlFlowDescriptor().getStatements();

  assert((lhsControlFlowStmts.size() == rhsControlFlowStmts.size()));
  for(int i = 0, size = lhsControlFlowStmts.size(); i < size; ++i) {
    if(!lhsControlFlowStmts[i]->ASTStmt->equals(rhsControlFlowStmts[i]->ASTStmt.get()))
      return false;
    if(lhsControlFlowStmts[i]->StackTrace) {
      if(rhsControlFlowStmts[i]->StackTrace) {
        for(int j = 0, jsize = lhsControlFlowStmts[i]->StackTrace->size(); j < jsize; ++j) {
          if(!(lhsControlFlowStmts[i]->StackTrace->at(j) ==
               rhsControlFlowStmts[i]->StackTrace->at(j))) {
            return false;
          }
        }
      }
      return false;
    }
  }

  return true;
}

int nextUID() {
  static int ret = 0;
  ret++;
  return ret;
} 

static void createCopyStencilIIRInMemory(std::shared_ptr<iir::StencilInstantiation>& target) {
  ///////////////// Generation of the IIR
  sir::Attr attributes;
  int stencilID = nextUID();
  target->getIIR()->insertChild(
      make_unique<iir::Stencil>(target->getMetaData(), attributes, stencilID), target->getIIR());
  const auto& IIRStencil = target->getIIR()->getChild(0);
  // One Multistage with a parallel looporder
  IIRStencil->insertChild(
      make_unique<iir::MultiStage>(target->getMetaData(), iir::LoopOrderKind::LK_Parallel));
  const auto& IIRMSS = (IIRStencil)->getChild(0);
  IIRMSS->setID(nextUID());

  // Create one stage inside the MSS
  IIRMSS->insertChild(make_unique<iir::Stage>(target->getMetaData(), nextUID()));
  const auto& IIRStage = IIRMSS->getChild(0);

  // Create one doMethod inside the Stage that spans the full domain
  IIRStage->insertChild(make_unique<iir::DoMethod>(
      iir::Interval(sir::Interval{sir::Interval::Start, sir::Interval::End}),
      target->getMetaData()));
  const auto& IIRDoMethod = IIRStage->getChild(0);
  IIRDoMethod->setID(nextUID());

  // create the StmtAccessPair
  auto sirInField = std::make_shared<sir::Field>("in_field");
  sirInField->IsTemporary = false;
  sirInField->fieldDimensions = Array3i{1, 1, 1};
  auto sirOutField = std::make_shared<sir::Field>("out_field");
  sirOutField->IsTemporary = false;
  sirOutField->fieldDimensions = Array3i{1, 1, 1};

  auto lhs = std::make_shared<ast::FieldAccessExpr>(sirOutField->Name);
  lhs->setID(nextUID());
  auto rhs = std::make_shared<ast::FieldAccessExpr>(sirInField->Name);
  rhs->setID(nextUID());

  int in_fieldID = target->getMetaData().insertField(iir::FieldAccessType::FAT_APIField,
                                                     sirInField->Name, sirInField->fieldDimensions);
  int out_fieldID = target->getMetaData().insertField(
      iir::FieldAccessType::FAT_APIField, sirOutField->Name, sirOutField->fieldDimensions);

  auto expr = std::make_shared<ast::AssignmentExpr>(lhs, rhs);
  expr->setID(nextUID());
  auto stmt = std::make_shared<ast::ExprStmt>(expr);
  stmt->setID(nextUID());
  auto statement = std::make_shared<Statement>(stmt, nullptr);
  auto insertee = make_unique<iir::StatementAccessesPair>(statement);
  // Add the accesses to the Pair:
  std::shared_ptr<iir::Accesses> callerAccesses = std::make_shared<iir::Accesses>();
  callerAccesses->addWriteExtent(out_fieldID, iir::Extents{0, 0, 0, 0, 0, 0});
  callerAccesses->addReadExtent(in_fieldID, iir::Extents{0, 0, 0, 0, 0, 0});
  insertee->setCallerAccesses(callerAccesses);
  // And add the StmtAccesspair to it
  IIRDoMethod->insertChild(std::move(insertee));
  IIRDoMethod->updateLevel();

  // Add the control flow descriptor to the IIR
  auto stencilCall = std::make_shared<ast::StencilCall>("generatedDriver");
  stencilCall->Args.push_back(sirInField->Name);
  stencilCall->Args.push_back(sirOutField->Name);
  auto placeholderStencil = std::make_shared<ast::StencilCall>(
      iir::InstantiationHelper::makeStencilCallCodeGenName(stencilID));
  auto stencilCallDeclStmt = std::make_shared<iir::StencilCallDeclStmt>(placeholderStencil);
  // Register the call and set it as a replacement for the next vertical region
  target->getMetaData().insertStencilCallStmt(stencilCallDeclStmt, stencilID);

  // auto stencilCallStmt = std::make_shared<ast::StencilCallDeclStmt>(stencilCall);
  // stencilCallStmt->setID(nextUID());
  // target->getMetaData().insertStencilCallStmt(stencilCallStmt, nextUID());
  auto stencilCallStatement = std::make_shared<Statement>(stencilCallDeclStmt, nullptr);
  target->getIIR()->getControlFlowDescriptor().insertStmt(stencilCallStatement);

  ///////////////// Generation of the Metadata

  target->getMetaData().setAccessIDNamePair(in_fieldID, "in_field");
  target->getMetaData().setAccessIDNamePair(out_fieldID, "out_field");
  target->getMetaData().insertExprToAccessID(lhs, out_fieldID);
  target->getMetaData().insertExprToAccessID(rhs, in_fieldID);
  target->getMetaData().setStencilname("generated");

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
}

static void readIIRFromFile(OptimizerContext &optimizer, std::shared_ptr<iir::StencilInstantiation>& target, std::string fname) {
  target =
      IIRSerializer::deserialize(fname, &optimizer, IIRSerializer::SK_Json);

  //this is whats actually to be tested.
  optimizer.restoreIIR("<restored>", target);
}

void deserialization_test_mat() {
  Options compileOptions;
  OptimizerContext::OptimizerContextOptions optimizerOptions;
  DawnCompiler compiler(&compileOptions); 
  OptimizerContext optimizer(compiler.getDiagnostics(), optimizerOptions, nullptr);

  //generate IIR in memory
  std::shared_ptr<iir::StencilInstantiation> copy_stencil_memory = std::make_shared<iir::StencilInstantiation>(&optimizer);
  createCopyStencilIIRInMemory(copy_stencil_memory);

  //read IIR from file
  std::shared_ptr<iir::StencilInstantiation> copy_stencil_from_file = std::make_shared<iir::StencilInstantiation>(&optimizer);
  readIIRFromFile(optimizer, copy_stencil_from_file, "copy_stencil_dumped.iir");

  printf("prepared IIRs succesfully!\n");

  if (compareIIRs(copy_stencil_memory->getIIR().get(), copy_stencil_from_file->getIIR().get())) {
    printf("IIRS are equal!\n");
  } else {
    printf("IIRS are different!\n");
  }
}