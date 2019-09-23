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

//this compares the structure of the iirs
static bool compareIIRstructures(iir::IIR* lhs, iir::IIR* rhs) {
  assert(lhs->checkTreeConsistency());
  assert(rhs->checkTreeConsistency());
  // checking the stencils
  assert(lhs->getChildren().size() == rhs->getChildren().size());
  for(int stencils = 0, size = lhs->getChildren().size(); stencils < size; ++stencils) {
    const auto& lhsStencil = lhs->getChild(stencils);
    const auto& rhsStencil = rhs->getChild(stencils);
    assert((lhsStencil->getStencilAttributes() == rhsStencil->getStencilAttributes()));    
    assert((lhsStencil->getStencilID() == rhsStencil->getStencilID()));

    // checking each of the multistages
    assert(lhsStencil->getChildren().size() == rhsStencil->getChildren().size());
    for(int mssidx = 0, mssSize = lhsStencil->getChildren().size(); mssidx < mssSize; ++mssidx) {
      const auto& lhsMSS = lhsStencil->getChild(mssidx);
      const auto& rhsMSS = rhsStencil->getChild(mssidx);
      assert((lhsMSS->getLoopOrder() == rhsMSS->getLoopOrder()));     
      assert((lhsMSS->getID() == rhsMSS->getID()));

      // checking each of the stages
      assert(lhsMSS->getChildren().size() == rhsMSS->getChildren().size());
      for(int stageidx = 0, stageSize = lhsMSS->getChildren().size(); stageidx < stageSize;
          ++stageidx) {
        const auto& lhsStage = lhsMSS->getChild(stageidx);
        const auto& rhsStage = rhsMSS->getChild(stageidx);
        assert((lhsStage->getStageID() == rhsStage->getStageID()));

        // checking each of the doMethods
        assert(lhsStage->getChildren().size() == rhsStage->getChildren().size());
        for(int doMethodidx = 0, doMethodSize = lhsStage->getChildren().size();
            doMethodidx < doMethodSize; ++doMethodidx) {
          const auto& lhsDoMethod = lhsStage->getChild(doMethodidx);
          const auto& rhsDoMethod = rhsStage->getChild(doMethodidx);
          assert((lhsDoMethod->getID() == rhsDoMethod->getID()));
          assert((lhsDoMethod->getInterval() == rhsDoMethod->getInterval()));

          // checking each of the StmtAccesspairs
          assert(lhsDoMethod->getChildren().size() == rhsDoMethod->getChildren().size());
          for(int stmtidx = 0, stmtSize = lhsDoMethod->getChildren().size(); stmtidx < stmtSize;
              ++stmtidx) {
            const auto& lhsStmt = lhsDoMethod->getChild(stmtidx);
            const auto& rhsStmt = rhsDoMethod->getChild(stmtidx);
            // check the statement
            assert(
                (lhsStmt->getStatement()->ASTStmt->equals(rhsStmt->getStatement()->ASTStmt.get())));

            // check the accesses
            assert(lhsStmt->getCallerAccesses()->getReadAccesses().size()
              == lhsStmt->getCallerAccesses()->getReadAccesses().size());
            assert(rhsStmt->getCallerAccesses()->getWriteAccesses().size()
              == rhsStmt->getCallerAccesses()->getWriteAccesses().size());

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

static bool compareMetaData(iir::StencilMetaInformation& lhs, iir::StencilMetaInformation& rhs) {
  assert((lhs.getExprIDToAccessIDMap() == rhs.getExprIDToAccessIDMap()));
  assert((lhs.getStmtIDToAccessIDMap() == rhs.getStmtIDToAccessIDMap()));
  assert((lhs.getAccessesOfType<iir::FieldAccessType::FAT_Literal>() ==
                  rhs.getAccessesOfType<iir::FieldAccessType::FAT_Literal>()));
  assert((lhs.getAccessesOfType<iir::FieldAccessType::FAT_Field>() ==
                  rhs.getAccessesOfType<iir::FieldAccessType::FAT_Field>()));
  assert((lhs.getAccessesOfType<iir::FieldAccessType::FAT_APIField>() ==
                  rhs.getAccessesOfType<iir::FieldAccessType::FAT_APIField>()));
  assert((lhs.getAccessesOfType<iir::FieldAccessType::FAT_StencilTemporary>() ==
                  rhs.getAccessesOfType<iir::FieldAccessType::FAT_StencilTemporary>()));
  assert((lhs.getAccessesOfType<iir::FieldAccessType::FAT_GlobalVariable>() ==
                  rhs.getAccessesOfType<iir::FieldAccessType::FAT_GlobalVariable>()));

  // we compare the content of the maps since the shared-ptr's are not the same
  assert((lhs.getFieldNameToBCMap().size() == rhs.getFieldNameToBCMap().size()));
  for(const auto& lhsPair : lhs.getFieldNameToBCMap()) {
    assert(rhs.getFieldNameToBCMap().count(lhsPair.first));
    auto rhsValue = rhs.getFieldNameToBCMap().at(lhsPair.first);
    assert(rhsValue->equals(lhsPair.second.get()));
  }
  assert((lhs.getFieldIDToDimsMap() == rhs.getFieldIDToDimsMap()));
  assert((lhs.getStencilLocation() == rhs.getStencilLocation()));
  assert((lhs.getStencilName() == rhs.getStencilName()));
  //file name makes little sense for in memory stencil
  //assert((lhs.getFileName() == rhs.getFileName()));  

  // we compare the content of the maps since the shared-ptr's are not the same
  assert((lhs.getStencilIDToStencilCallMap().getDirectMap().size() ==
                  rhs.getStencilIDToStencilCallMap().getDirectMap().size()));
  for(const auto& lhsPair : lhs.getStencilIDToStencilCallMap().getDirectMap()) {
    assert(rhs.getStencilIDToStencilCallMap().getDirectMap().count(lhsPair.first));
    auto rhsValue = rhs.getStencilIDToStencilCallMap().getDirectMap().at(lhsPair.first);
    assert(rhsValue->equals(lhsPair.second.get()));
  }

  return true;
}

static bool compareDerivedInformation(iir::IIR* lhs, iir::IIR* rhs) {
  //IIR -> Stencil -> Multistage -> Stage -> Do Method -> (StatementAccessPair)

  //IIR
  // struct DerivedInfo {
  //   /// StageID to name Map. Filled by the `PassSetStageName`.
  //   [X] std::unordered_map<int, std::string> StageIDToNameMap_;
  //   /// Set containing the AccessIDs of fields which are manually allocated by the stencil and serve
  //   /// as temporaries spanning over multiple stencils
  // };

  // assert(lhs->getStageIDToNameMap() == rhs->getStageIDToNameMap());  //this is not present in stencil from memory

  for(int stencils = 0, size = lhs->getChildren().size(); stencils < size; ++stencils) {
    const auto& lhsStencil = lhs->getChild(stencils);
    const auto& rhsStencil = rhs->getChild(stencils);    

    //struct DerivedInfo {
    //   /// Dependency graph of the stages of this stencil
    //   [X] std::shared_ptr<DependencyGraphStage> stageDependencyGraph_;
    //   /// field info properties
    //   [X] std::unordered_map<int, FieldInfo> fields_;  
    // };

    assert(lhsStencil->getStageDependencyGraph() == rhsStencil->getStageDependencyGraph()); //this _should_ work, to be tested
    assert(lhsStencil->getFields() == rhsStencil->getFields());

    // checking each of the multistages
    for(int mssidx = 0, mssSize = lhsStencil->getChildren().size(); mssidx < mssSize; ++mssidx) {
      const auto& lhsMSS = lhsStencil->getChild(mssidx);
      const auto& rhsMSS = rhsStencil->getChild(mssidx);

      assert((lhsMSS->getCaches().size() == rhsMSS->getCaches().size()));
      for(const auto& lhsPair : lhsMSS->getCaches()) {
        assert(rhsMSS->getCaches().count(lhsPair.first));
        auto rhsValue = rhsMSS->getCaches().at(lhsPair.first);
        assert((rhsValue == lhsPair.second));
      }

      assert(lhsMSS->getFields() == rhsMSS->getFields());

      // checking each of the stages
      for(int stageidx = 0, stageSize = lhsMSS->getChildren().size(); stageidx < stageSize;
          ++stageidx) {
        const auto& lhsStage = lhsMSS->getChild(stageidx);
        const auto& rhsStage = rhsMSS->getChild(stageidx);

        // struct DerivedInfo {

        //   /// Declaration of the fields of this stage
        //   [X] std::unordered_map<int, Field> fields_;

        //   /// AccessIDs of the global variable accesses of this stage
        //   [X] std::unordered_set<int> allGlobalVariables_;
        //   [X] std::unordered_set<int> globalVariables_;
        //   [X] std::unordered_set<int> globalVariablesFromStencilFunctionCalls_;

        //   [X] Extents extents_;
        //   [X] bool requiresSync_ = false;
        // };

        assert(lhsStage->getFields() == rhsStage->getFields());
        assert(lhsStage->getAllGlobalVariables() == rhsStage->getAllGlobalVariables());
        assert(lhsStage->getGlobalVariables() == rhsStage->getGlobalVariables());
        assert(lhsStage->getGlobalVariablesFromStencilFunctionCalls() 
          == rhsStage->getGlobalVariablesFromStencilFunctionCalls());
        assert(lhsStage->getExtents() == rhsStage->getExtents());
        assert(lhsStage->getRequiresSync() == rhsStage->getRequiresSync());

        // checking each of the doMethods
        for(int doMethodidx = 0, doMethodSize = lhsStage->getChildren().size();
            doMethodidx < doMethodSize; ++doMethodidx) {
          const auto& lhsDoMethod = lhsStage->getChild(doMethodidx);
          const auto& rhsDoMethod = rhsStage->getChild(doMethodidx);

          // struct DerivedInfo {
          //   /// Declaration of the fields of this doMethod
          //   [X] std::unordered_map<int, Field> fields_;
          //   [X] std::shared_ptr<DependencyGraphAccesses> dependencyGraph_;
          // };

          assert(lhsDoMethod->getFields() == rhsDoMethod->getFields());
          assert(lhsDoMethod->getDependencyGraph() == rhsDoMethod->getDependencyGraph());
        }
      }
    }
  }

  return true;
}

static void createCopyStencilIIRInMemory(std::shared_ptr<iir::StencilInstantiation>& target) {
  ///////////////// Generation of the IIR
  sir::Attr attributes;
  int stencilID = target->nextUID();
  target->getIIR()->insertChild(
      make_unique<iir::Stencil>(target->getMetaData(), attributes, stencilID), target->getIIR());
  const auto& IIRStencil = target->getIIR()->getChild(0);
  // One Multistage with a parallel looporder
  IIRStencil->insertChild(
      make_unique<iir::MultiStage>(target->getMetaData(), iir::LoopOrderKind::LK_Parallel));
  const auto& IIRMSS = (IIRStencil)->getChild(0);
  IIRMSS->setID(target->nextUID());

  // Create one stage inside the MSS
  IIRMSS->insertChild(make_unique<iir::Stage>(target->getMetaData(), target->nextUID()));
  const auto& IIRStage = IIRMSS->getChild(0);

  // Create one doMethod inside the Stage that spans the full domain
  IIRStage->insertChild(make_unique<iir::DoMethod>(
      iir::Interval(sir::Interval{sir::Interval::Start, sir::Interval::End}),
      target->getMetaData()));
  const auto& IIRDoMethod = IIRStage->getChild(0);
  IIRDoMethod->setID(target->nextUID());

  // create the StmtAccessPair
  auto sirInField = std::make_shared<sir::Field>("in_field");
  sirInField->IsTemporary = false;
  sirInField->fieldDimensions = Array3i{1, 1, 1};
  auto sirOutField = std::make_shared<sir::Field>("out_field");
  sirOutField->IsTemporary = false;
  sirOutField->fieldDimensions = Array3i{1, 1, 1};

  auto lhs = std::make_shared<ast::FieldAccessExpr>(sirOutField->Name);
  lhs->setID(target->nextUID());
  auto rhs = std::make_shared<ast::FieldAccessExpr>(sirInField->Name);
  rhs->setID(target->nextUID());

  int in_fieldID = target->getMetaData().addField(iir::FieldAccessType::FAT_APIField,
                                                     sirInField->Name, sirInField->fieldDimensions);
  int out_fieldID = target->getMetaData().addField(
      iir::FieldAccessType::FAT_APIField, sirOutField->Name, sirOutField->fieldDimensions);

  auto expr = std::make_shared<ast::AssignmentExpr>(lhs, rhs);
  expr->setID(target->nextUID());
  auto stmt = std::make_shared<ast::ExprStmt>(expr);
  stmt->setID(target->nextUID());
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
  target->getMetaData().addStencilCallStmt(stencilCallDeclStmt, stencilID);

  auto stencilCallStatement = std::make_shared<Statement>(stencilCallDeclStmt, nullptr);
  target->getIIR()->getControlFlowDescriptor().insertStmt(stencilCallStatement);

  ///////////////// Generation of the Metadata

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

static void createLapStencilIIRInMemory(std::shared_ptr<iir::StencilInstantiation>& target) {
  ///////////////// Generation of the IIR
  sir::Attr attributes;
  int stencilID = target->nextUID();
  target->getIIR()->insertChild(
      make_unique<iir::Stencil>(target->getMetaData(), attributes, stencilID), target->getIIR());
  const auto& IIRStencil = target->getIIR()->getChild(0);
  // One Multistage with a parallel looporder
  IIRStencil->insertChild(
      make_unique<iir::MultiStage>(target->getMetaData(), iir::LoopOrderKind::LK_Parallel));
  const auto& IIRMSS = (IIRStencil)->getChild(0);
  IIRMSS->setID(target->nextUID());

  auto IIRStage1 = make_unique<iir::Stage>(target->getMetaData(), target->nextUID());
  auto IIRStage2 = make_unique<iir::Stage>(target->getMetaData(), target->nextUID());

  IIRStage1->setExtents(iir::Extents(-1, +1, -1, +1, 0, 0));

  // Create one doMethod inside the Stage that spans the full domain
  IIRStage1->insertChild(make_unique<iir::DoMethod>(
      iir::Interval(sir::Interval{sir::Interval::Start, sir::Interval::End}),
      target->getMetaData()));
  const auto& IIRDoMethod1 = IIRStage1->getChild(0);
  IIRDoMethod1->setID(target->nextUID());

  IIRStage2->insertChild(make_unique<iir::DoMethod>(
      iir::Interval(sir::Interval{sir::Interval::Start, sir::Interval::End}),
      target->getMetaData()));
  const auto& IIRDoMethod2 = IIRStage2->getChild(0);
  IIRDoMethod2->setID(target->nextUID());

  // Create two stages inside the MSS
  IIRMSS->insertChild(std::move(IIRStage1));
  IIRMSS->insertChild(std::move(IIRStage2));

  // create the StmtAccessPair
  auto sirInField = std::make_shared<sir::Field>("in");
  sirInField->IsTemporary = false;
  sirInField->fieldDimensions = Array3i{1, 1, 1};
  auto sirOutField = std::make_shared<sir::Field>("out");
  sirOutField->IsTemporary = false;
  sirOutField->fieldDimensions = Array3i{1, 1, 1};
  auto sirTmpField = std::make_shared<sir::Field>("tmp");
  sirOutField->IsTemporary = true;
  sirOutField->fieldDimensions = Array3i{1, 1, 1};

  auto lhsTmp = std::make_shared<ast::FieldAccessExpr>(sirTmpField->Name);
  lhsTmp->setID(target->nextUID());
  
  auto rhsInT1 = std::make_shared<ast::FieldAccessExpr>(sirInField->Name, Array3i{0, -2, 0});
  auto rhsInT2 = std::make_shared<ast::FieldAccessExpr>(sirInField->Name, Array3i{0, +2, 0});
  auto rhsInT3 = std::make_shared<ast::FieldAccessExpr>(sirInField->Name, Array3i{-2, 0, 0});
  auto rhsInT4 = std::make_shared<ast::FieldAccessExpr>(sirInField->Name, Array3i{+2, 0, 0});

  rhsInT1->setID(target->nextUID());
  rhsInT2->setID(target->nextUID());
  rhsInT3->setID(target->nextUID());
  rhsInT4->setID(target->nextUID());

  auto lhsOut = std::make_shared<ast::FieldAccessExpr>(sirOutField->Name);
  lhsOut->setID(target->nextUID());

  auto rhsTmpT1 = std::make_shared<ast::FieldAccessExpr>(sirTmpField->Name, Array3i{0, -1, 0});
  auto rhsTmpT2 = std::make_shared<ast::FieldAccessExpr>(sirTmpField->Name, Array3i{0, +1, 0});
  auto rhsTmpT3 = std::make_shared<ast::FieldAccessExpr>(sirTmpField->Name, Array3i{-1, 0, 0});
  auto rhsTmpT4 = std::make_shared<ast::FieldAccessExpr>(sirTmpField->Name, Array3i{+1, 0, 0});

  rhsTmpT1->setID(target->nextUID());
  rhsTmpT2->setID(target->nextUID());
  rhsTmpT3->setID(target->nextUID());
  rhsTmpT4->setID(target->nextUID());

  int inFieldID = target->getMetaData().addField(
    iir::FieldAccessType::FAT_APIField, sirInField->Name, sirInField->fieldDimensions);
  int tmpFieldID = target->getMetaData().addField(
      iir::FieldAccessType::FAT_StencilTemporary, sirTmpField->Name, sirTmpField->fieldDimensions);
  int outFieldID = target->getMetaData().addField(
      iir::FieldAccessType::FAT_APIField, sirOutField->Name, sirOutField->fieldDimensions);

  auto plusIn1 = std::make_shared<ast::BinaryOperator>(rhsInT1, std::string("+"), rhsInT2);
  auto plusIn2 = std::make_shared<ast::BinaryOperator>(rhsInT3, std::string("+"), rhsInT4);
  auto plusIn3 = std::make_shared<ast::BinaryOperator>(plusIn1, std::string("+"), plusIn2);

  plusIn1->setID(target->nextUID());
  plusIn2->setID(target->nextUID());
  plusIn3->setID(target->nextUID());

  auto assignmentTmpIn = std::make_shared<ast::AssignmentExpr>(lhsTmp, plusIn3);
  assignmentTmpIn->setID(target->nextUID());

  auto stmt1 = std::make_shared<ast::ExprStmt>(assignmentTmpIn);
  stmt1->setID(target->nextUID());
  auto statement1 = std::make_shared<Statement>(stmt1, nullptr);
  auto insertee1 = make_unique<iir::StatementAccessesPair>(statement1);
  // Add the accesses to the Pair:
  std::shared_ptr<iir::Accesses> callerAccesses1 = std::make_shared<iir::Accesses>();
  callerAccesses1->addWriteExtent(tmpFieldID, iir::Extents{0, 0, 0, 0, 0, 0});
  callerAccesses1->addReadExtent(inFieldID, iir::Extents{-2, 2, -2, 2, 0, 0});
  insertee1->setCallerAccesses(callerAccesses1);
  // And add the StmtAccesspair to it
  IIRDoMethod1->insertChild(std::move(insertee1));
  IIRDoMethod1->updateLevel();

  auto plusTmp1 = std::make_shared<ast::BinaryOperator>(rhsTmpT1, std::string("+"), rhsTmpT2);
  auto plusTmp2 = std::make_shared<ast::BinaryOperator>(rhsTmpT3, std::string("+"), rhsTmpT4);
  auto plusTmp3 = std::make_shared<ast::BinaryOperator>(plusTmp1, std::string("+"), plusTmp2);

  plusTmp1->setID(target->nextUID());
  plusTmp2->setID(target->nextUID());
  plusTmp3->setID(target->nextUID());

  auto assignmentOutTmp = std::make_shared<ast::AssignmentExpr>(lhsOut, plusTmp3);
  assignmentOutTmp->setID(target->nextUID());

  auto stmt2 = std::make_shared<ast::ExprStmt>(assignmentOutTmp);
  stmt2->setID(target->nextUID());
  auto statement2 = std::make_shared<Statement>(stmt2, nullptr);
  auto insertee2 = make_unique<iir::StatementAccessesPair>(statement2);
  // Add the accesses to the Pair:
  std::shared_ptr<iir::Accesses> callerAccesses2 = std::make_shared<iir::Accesses>();
  callerAccesses2->addWriteExtent(outFieldID, iir::Extents{0, 0, 0, 0, 0, 0});
  callerAccesses2->addReadExtent(tmpFieldID, iir::Extents{-1, 1, -1, 1, 0, 0});
  insertee2->setCallerAccesses(callerAccesses2);
  // And add the StmtAccesspair to it
  IIRDoMethod2->insertChild(std::move(insertee2));
  IIRDoMethod2->updateLevel();

  // Add the control flow descriptor to the IIR
  auto stencilCall = std::make_shared<ast::StencilCall>("generatedDriver");
  stencilCall->Args.push_back(sirInField->Name);
  // stencilCall->Args.push_back(sirTmpField->Name);
  stencilCall->Args.push_back(sirOutField->Name);
  auto placeholderStencil = std::make_shared<ast::StencilCall>(
      iir::InstantiationHelper::makeStencilCallCodeGenName(stencilID));
  auto stencilCallDeclStmt = std::make_shared<iir::StencilCallDeclStmt>(placeholderStencil);
  // Register the call and set it as a replacement for the next vertical region
  target->getMetaData().addStencilCallStmt(stencilCallDeclStmt, stencilID);

  auto stencilCallStatement = std::make_shared<Statement>(stencilCallDeclStmt, nullptr);
  target->getIIR()->getControlFlowDescriptor().insertStmt(stencilCallStatement);

  ///////////////// Generation of the Metadata

  target->getMetaData().insertExprToAccessID(lhsTmp, tmpFieldID);
  target->getMetaData().insertExprToAccessID(rhsInT1, inFieldID);
  target->getMetaData().insertExprToAccessID(rhsInT2, inFieldID);
  target->getMetaData().insertExprToAccessID(rhsInT3, inFieldID);
  target->getMetaData().insertExprToAccessID(rhsInT4, inFieldID);

  target->getMetaData().insertExprToAccessID(lhsOut, outFieldID);
  target->getMetaData().insertExprToAccessID(rhsTmpT1, tmpFieldID);
  target->getMetaData().insertExprToAccessID(rhsTmpT2, tmpFieldID);
  target->getMetaData().insertExprToAccessID(rhsTmpT3, tmpFieldID);
  target->getMetaData().insertExprToAccessID(rhsTmpT4, tmpFieldID);

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

static void compareIIRs(std::shared_ptr<iir::StencilInstantiation> lhs, std::shared_ptr<iir::StencilInstantiation> rhs) {
  //first compare the (structure of the) iirs, this is a precondition before we can actually check the metadata / derived info
  if (compareIIRstructures(lhs->getIIR().get(), rhs->getIIR().get())) {
    printf("IIRS are equal!\n");
  } else {
    printf("IIRS are different!\n");
  }

  //then we compare the meta data
  if (compareMetaData(lhs->getMetaData(), rhs->getMetaData())) {
    printf("Meta Data is equal!\n");
  } else {
    printf("Meta Data is different!\n");
  }

  //and finally the derived info 
  if (compareDerivedInformation(lhs->getIIR().get(), rhs->getIIR().get())) {
    printf("Derived Info is equal!\n");
  } else {
    printf("Derived Info is different!\n");
  }
}

void deserialization_test_mat() {
  Options compileOptions;
  OptimizerContext::OptimizerContextOptions optimizerOptions;
  DawnCompiler compiler(&compileOptions);   
  OptimizerContext optimizer(compiler.getDiagnostics(), optimizerOptions, std::make_shared<dawn::SIR>());

  //read IIRs from file
  std::shared_ptr<iir::StencilInstantiation> copy_stencil_from_file = std::make_shared<iir::StencilInstantiation>(
    *optimizer.getSIR()->GlobalVariableMap, optimizer.getSIR()->StencilFunctions);
  readIIRFromFile(optimizer, copy_stencil_from_file, "test_iirs/copy_stencil.iir");
  UIDGenerator::getInstance()->reset();

  std::shared_ptr<iir::StencilInstantiation> lap_stencil_from_file = std::make_shared<iir::StencilInstantiation>(
    *optimizer.getSIR()->GlobalVariableMap, optimizer.getSIR()->StencilFunctions);
  readIIRFromFile(optimizer, lap_stencil_from_file, "test_iirs/lap_stencil.iir");
  UIDGenerator::getInstance()->reset();

  //generate IIRs in memory
  std::shared_ptr<iir::StencilInstantiation> copy_stencil_memory = std::make_shared<iir::StencilInstantiation>(
    *optimizer.getSIR()->GlobalVariableMap, optimizer.getSIR()->StencilFunctions);
  createCopyStencilIIRInMemory(copy_stencil_memory);
  UIDGenerator::getInstance()->reset();
  // IIRSerializer::serialize("copy_stencil.iir", copy_stencil_memory, IIRSerializer::SK_Json);

  std::shared_ptr<iir::StencilInstantiation> lap_stencil_memory = std::make_shared<iir::StencilInstantiation>(   
    *optimizer.getSIR()->GlobalVariableMap, optimizer.getSIR()->StencilFunctions);
  createLapStencilIIRInMemory(lap_stencil_memory); 
  UIDGenerator::getInstance()->reset();
  IIRSerializer::serialize("lap_stencil.iir", lap_stencil_memory, IIRSerializer::SK_Json);

  printf("prepared IIRs succesfully!\n");

  compareIIRs(copy_stencil_from_file, copy_stencil_memory);
  compareIIRs(lap_stencil_from_file, lap_stencil_memory);
}