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

#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/AST/ASTExpr.h"
#include "dawn/AST/ASTStmt.h"
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/ASTUtil.h"
#include "dawn/IIR/FieldAccessMetadata.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/InstantiationHelper.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/AccessComputation.h"
#include "dawn/Optimizer/PassTemporaryType.h"
#include "dawn/Optimizer/StatementMapper.h"
#include "dawn/SIR/ASTFwd.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/STLExtras.h"
#include <stack>

#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include "dawn/Support/Type.h"
#include <memory>
#include <string>

namespace dawn {

static void createIIRInMemory(std::shared_ptr<iir::StencilInstantiation>& target) {
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

  // create the SIR-Fields
  auto sirInField = std::make_shared<sir::Field>("in_field");
  sirInField->IsTemporary = false;
  sirInField->fieldDimensions = Array3i{1, 1, 1};
  auto sirOutField = std::make_shared<sir::Field>("out_field");
  sirOutField->IsTemporary = false;
  sirOutField->fieldDimensions = Array3i{1, 1, 1};
  int in_fieldID = target->getMetaData().insertField(iir::FieldAccessType::FAT_APIField,
                                                     sirInField->Name, sirInField->fieldDimensions);
  int out_fieldID = target->getMetaData().insertField(
      iir::FieldAccessType::FAT_APIField, sirOutField->Name, sirOutField->fieldDimensions);

  int literal_m3_ID =
      target->getMetaData().insertAccessOfType(iir::FieldAccessType::FAT_Literal, "-3");
  int literal_d1_ID =
      target->getMetaData().insertAccessOfType(iir::FieldAccessType::FAT_Literal, "0.1");
  // create the StmtAccessPair 1:
  // m_out_field[t] = -3. * m_in_field[t];
  //==----------------------------------------------------------------------------------------------
  // create the stmt
  auto outFieldAccess = std::make_shared<ast::FieldAccessExpr>(sirOutField->Name);
  outFieldAccess->setID(target->nextUID());
  auto inFieldAccess = std::make_shared<ast::FieldAccessExpr>(sirInField->Name);
  inFieldAccess->setID(target->nextUID());
  auto literalAccess = std::make_shared<ast::LiteralAccessExpr>("-3", dawn::BuiltinTypeID::Float);
  literalAccess->setID(target->nextUID());
  auto binop = std::make_shared<ast::BinaryOperator>(literalAccess, "*", inFieldAccess);
  binop->setID(target->nextUID());
  auto assignment = std::make_shared<ast::AssignmentExpr>(outFieldAccess, binop);
  assignment->setID(target->nextUID());

  // Insert the stmt into the statementaccesspair
  auto scalingInputES = std::make_shared<iir::ExprStmt>(assignment);
  auto scalingInput = std::make_shared<Statement>(scalingInputES, nullptr);
  auto sap_1 = make_unique<iir::StatementAccessesPair>(scalingInput);
  std::shared_ptr<iir::Accesses> callerAccesses_1 = std::make_shared<iir::Accesses>();
  callerAccesses_1->addWriteExtent(out_fieldID, iir::Extents{0, 0, 0, 0, 0, 0});
  callerAccesses_1->addReadExtent(in_fieldID, iir::Extents{0, 0, 0, 0, 0, 0});
  sap_1->setCallerAccesses(callerAccesses_1);

  // Add the statementaccesspair to the IIR
  IIRDoMethod->insertChild(std::move(sap_1));
  IIRDoMethod->updateLevel();
  //==----------------------------------------------------------------------------------------------

  // create the StmtAccessPair 2:
  // for (auto&& x : cellNeighboursOfCell(m_mesh, t)) m_out_field[t] += m_in_field[*x];
  //==----------------------------------------------------------------------------------------------
  // create the stmt
  auto lhs = std::make_shared<ast::FieldAccessExpr>(sirOutField->Name);
  lhs->setID(target->nextUID());
  auto rhs = std::make_shared<ast::FieldAccessExpr>(sirInField->Name);
  rhs->setID(target->nextUID());
  auto stmt = std::make_shared<ast::ReductionOverNeighborStmt>(lhs, "+", rhs);
  stmt->setID(target->nextUID());
  auto statement = std::make_shared<Statement>(stmt, nullptr);
  auto insertee = make_unique<iir::StatementAccessesPair>(statement);

  // Insert the stmt into the statementaccesspair
  std::shared_ptr<iir::Accesses> callerAccesses = std::make_shared<iir::Accesses>();
  callerAccesses->addWriteExtent(out_fieldID, iir::Extents{0, 0, 0, 0, 0, 0});
  callerAccesses->addReadExtent(in_fieldID, iir::Extents{0, 0, 0, 0, 0, 0});
  insertee->setCallerAccesses(callerAccesses);

  // Add the statementaccesspair to the IIR
  IIRDoMethod->insertChild(std::move(insertee));
  IIRDoMethod->updateLevel();
  //==----------------------------------------------------------------------------------------------

  // create the StmtAccessPair 3:
  // m_out_field[x] *= 0.1;
  //==----------------------------------------------------------------------------------------------
  // create the stmt
  auto outAccess = std::make_shared<ast::FieldAccessExpr>(sirOutField->Name);
  outAccess->setID(target->nextUID());
  auto diffCoeffAccess =
      std::make_shared<ast::LiteralAccessExpr>("0.3", dawn::BuiltinTypeID::Float);
  diffCoeffAccess->setID(target->nextUID());
  auto scale = std::make_shared<ast::AssignmentExpr>(outAccess, diffCoeffAccess, "*=");
  scale->setID(target->nextUID());

  // Insert the stmt into the statementaccesspair
  auto scaleOutput = std::make_shared<Statement>(std::make_shared<ast::ExprStmt>(scale), nullptr);
  auto sap_2 = make_unique<iir::StatementAccessesPair>(scaleOutput);
  std::shared_ptr<iir::Accesses> callerAccesses_2 = std::make_shared<iir::Accesses>();
  callerAccesses_2->addWriteExtent(out_fieldID, iir::Extents{0, 0, 0, 0, 0, 0});
  callerAccesses_2->addReadExtent(in_fieldID, iir::Extents{0, 0, 0, 0, 0, 0});
  sap_2->setCallerAccesses(callerAccesses_2);

  // Add the statementaccesspair to the IIR
  IIRDoMethod->insertChild(std::move(sap_2));
  IIRDoMethod->updateLevel();
  //==----------------------------------------------------------------------------------------------

  // create the StmtAccessPair 4:
  // m_out_field[x] += m_in_field[x];
  //==----------------------------------------------------------------------------------------------
  // create the stmt
  auto finalOutAccess = std::make_shared<ast::FieldAccessExpr>(sirOutField->Name);
  finalOutAccess->setID(target->nextUID());
  auto inAccess = std::make_shared<ast::FieldAccessExpr>(sirInField->Name);
  inAccess->setID(target->nextUID());
  auto addUp = std::make_shared<ast::AssignmentExpr>(finalOutAccess, inAccess, "+=");
  addUp->setID(target->nextUID());

  // Insert the stmt into the statementaccesspair
  auto addOld = std::make_shared<Statement>(std::make_shared<ast::ExprStmt>(addUp), nullptr);
  auto sap_3 = make_unique<iir::StatementAccessesPair>(addOld);
  std::shared_ptr<iir::Accesses> callerAccesses_3 = std::make_shared<iir::Accesses>();
  callerAccesses_3->addWriteExtent(out_fieldID, iir::Extents{0, 0, 0, 0, 0, 0});
  callerAccesses_3->addReadExtent(in_fieldID, iir::Extents{0, 0, 0, 0, 0, 0});
  sap_3->setCallerAccesses(callerAccesses_3);

  // Add the statementaccesspair to the IIR
  IIRDoMethod->insertChild(std::move(sap_3));
  IIRDoMethod->updateLevel();
  //==----------------------------------------------------------------------------------------------

  // Add the control flow descriptor to the IIR
  auto stencilCall = std::make_shared<sir::StencilCall>("generatedDriver");
  stencilCall->Args.push_back(sirInField);
  stencilCall->Args.push_back(sirOutField);
  auto placeholderStencil = std::make_shared<sir::StencilCall>(
      iir::InstantiationHelper::makeStencilCallCodeGenName(stencilID));
  auto stencilCallDeclStmt = std::make_shared<iir::StencilCallDeclStmt>(placeholderStencil);
  // Register the call and set it as a replacement for the next vertical region
  target->getMetaData().insertStencilCallStmt(stencilCallDeclStmt, stencilID);

  // auto stencilCallStmt = std::make_shared<ast::StencilCallDeclStmt>(stencilCall);
  // stencilCallStmt->setID(target->nextUID());
  // target->getMetaData().insertStencilCallStmt(stencilCallStmt, target->nextUID());
  auto stencilCallStatement = std::make_shared<Statement>(stencilCallDeclStmt, nullptr);
  target->getIIR()->getControlFlowDescriptor().insertStmt(stencilCallStatement);

  ///////////////// Generation of the Metadata

  target->getMetaData().setAccessIDNamePair(in_fieldID, "in_field");
  target->getMetaData().setAccessIDNamePair(out_fieldID, "out_field");

  // stmt 1
  target->getMetaData().insertExprToAccessID(outFieldAccess, out_fieldID);
  target->getMetaData().insertExprToAccessID(inFieldAccess, in_fieldID);
  target->getMetaData().insertExprToAccessID(literalAccess, literal_m3_ID);

  // stmt 2
  target->getMetaData().insertExprToAccessID(lhs, out_fieldID);
  target->getMetaData().insertExprToAccessID(rhs, in_fieldID);

  // stmt 3
  target->getMetaData().insertExprToAccessID(outAccess, out_fieldID);
  target->getMetaData().insertExprToAccessID(diffCoeffAccess, literal_d1_ID);

  // stmt 4
  target->getMetaData().insertExprToAccessID(finalOutAccess, out_fieldID);
  target->getMetaData().insertExprToAccessID(inAccess, in_fieldID);

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
namespace {
using namespace iir;
//===------------------------------------------------------------------------------------------===//
//     StencilDescStatementMapper
//===------------------------------------------------------------------------------------------===//

/// @brief Map the statements of the stencil description AST to a flat list of statements and
/// inline all calls to other stencils
class StencilDescStatementMapper : public iir::ASTVisitor {

  /// @brief Record of the current scope (each StencilCall will create a new scope)
  struct Scope : public NonCopyable {
    Scope(const std::string& name, ControlFlowDescriptor& controlFlowDescriptor)
        : Name(name), ScopeDepth(0), controlFlowDescriptor_(controlFlowDescriptor),
          StackTrace(nullptr) {}

    /// Name of the current stencil
    std::string Name;

    /// Nesting of scopes
    int ScopeDepth;

    /// List of statements of the stencil description
    ControlFlowDescriptor& controlFlowDescriptor_;

    /// Scope fieldnames to to (global) AccessID
    std::unordered_map<std::string, int> LocalFieldnameToAccessIDMap;

    /// Scope variable name to (global) AccessID
    std::unordered_map<std::string, int> LocalVarNameToAccessIDMap;

    /// Map of known values of variables
    std::unordered_map<std::string, double> VariableMap;

    /// Current call stack of stencil calls (may be NULL)
    std::shared_ptr<std::vector<sir::StencilCall*>> StackTrace;
  };

  const std::shared_ptr<iir::StencilInstantiation>& instantiation_;
  iir::StencilMetaInformation& metadata_;
  std::stack<std::shared_ptr<Scope>> scope_;

  sir::Stencil* sirStencil_;

  const std::shared_ptr<SIR> sir_;

  /// We replace the first VerticalRegionDeclStmt with a dummy node which signals code-gen that it
  /// should insert a call to the gridtools stencil here
  std::shared_ptr<iir::Stmt> stencilDescReplacement_;

public:
  StencilDescStatementMapper(std::shared_ptr<iir::StencilInstantiation>& instantiation,
                             sir::Stencil* sirStencil, const std::shared_ptr<SIR>& sir)
      : instantiation_(instantiation), metadata_(instantiation->getMetaData()),
        sirStencil_(sirStencil), sir_(sir) {
    DAWN_ASSERT(instantiation);
    // Create the initial scope
    scope_.push(std::make_shared<Scope>(sirStencil_->Name,
                                        instantiation_->getIIR()->getControlFlowDescriptor()));
    scope_.top()->LocalFieldnameToAccessIDMap = metadata_.getNameToAccessIDMap();

    // We add all global variables which have constant values
    for(auto& keyValuePair : *(sir->GlobalVariableMap)) {
      const std::string& key = keyValuePair.first;
      sir::Value& value = *keyValuePair.second;

      if(value.isConstexpr()) {
        switch(value.getType()) {
        case sir::Value::Boolean:
          scope_.top()->VariableMap[key] = value.getValue<bool>();
          break;
        case sir::Value::Integer:
          scope_.top()->VariableMap[key] = value.getValue<int>();
          break;
        case sir::Value::Double:
          scope_.top()->VariableMap[key] = value.getValue<double>();
          break;
        default:
          break;
        }
      }
    }
    // We start with a single stencil
    makeNewStencil();
  }

  /// @brief Create a new stencil in the instantiation and prepare the replacement node for the
  /// next VerticalRegionDeclStmt
  /// @see tryReplaceVerticalRegionDeclStmt
  void makeNewStencil() {
    int StencilID = instantiation_->nextUID();
    instantiation_->getIIR()->insertChild(
        make_unique<Stencil>(metadata_, sirStencil_->Attributes, StencilID),
        instantiation_->getIIR());
    // We create a paceholder stencil-call for CodeGen to know wehere we need to insert calls to
    // this stencil
    auto placeholderStencil = std::make_shared<sir::StencilCall>(
        InstantiationHelper::makeStencilCallCodeGenName(StencilID));
    auto stencilCallDeclStmt = std::make_shared<iir::StencilCallDeclStmt>(placeholderStencil);

    // Register the call and set it as a replacement for the next vertical region
    metadata_.insertStencilCallStmt(stencilCallDeclStmt, StencilID);
    stencilDescReplacement_ = stencilCallDeclStmt;
  }

  /// @brief Replace the first VerticalRegionDeclStmt or StencilCallDelcStmt with a dummy
  /// placeholder signaling code-gen that it should insert a call to the gridtools stencil.
  ///
  /// All remaining VerticalRegion/StencilCalls statements which are still in the stencil
  /// description AST are pruned at the end
  ///
  /// @see removeObsoleteStencilDescNodes
  void tryReplaceStencilDescStmt(const std::shared_ptr<iir::Stmt>& stencilDescNode) {
    DAWN_ASSERT(stencilDescNode->isStencilDesc());

    // Nothing to do, statement was already replaced
    if(!stencilDescReplacement_)
      return;

    // TODO redo
    // Instead of inserting the VerticalRegionDeclStmt we insert the call to the gridtools stencil
    if(scope_.top()->ScopeDepth == 1)
      scope_.top()->controlFlowDescriptor_.insertStmt(
          std::make_shared<Statement>(stencilDescReplacement_, scope_.top()->StackTrace));
    else {

      // We need to replace the VerticalRegionDeclStmt in the current statement
      iir::replaceOldStmtWithNewStmtInStmt(
          scope_.top()->controlFlowDescriptor_.getStatements().back()->ASTStmt, stencilDescNode,
          stencilDescReplacement_);
    }

    stencilDescReplacement_ = nullptr;
  }

  /// @brief Remove all VerticalRegionDeclStmt and StencilCallDeclStmt (which do not start with
  /// `GridToolsStencilCallPrefix`) from the list of statements and remove empty stencils
  void cleanupStencilDeclAST() {

    // We only need to remove "nested" nodes as the top-level VerticalRegions or StencilCalls are
    // not inserted into the statement list in the frist place
    class RemoveStencilDescNodes : public iir::ASTVisitorForwarding {
    public:
      RemoveStencilDescNodes() {}

      bool needsRemoval(const std::shared_ptr<iir::Stmt>& stmt) const {
        if(StencilCallDeclStmt* s = dyn_cast<iir::StencilCallDeclStmt>(stmt.get())) {
          // StencilCallDeclStmt node, remove it if it is not one of our artificial stencil call
          // nodes
          if(!InstantiationHelper::isStencilCallCodeGenName(s->getStencilCall()->Callee))
            return true;
        } else if(isa<iir::VerticalRegionDeclStmt>(stmt.get())) {
          // Remove all remaining vertical regions
          return true;
        }

        return false;
      }

      void visit(const std::shared_ptr<iir::BlockStmt>& stmt) override {
        for(auto it = stmt->getStatements().begin(); it != stmt->getStatements().end();) {
          if(needsRemoval(*it)) {
            it = stmt->getStatements().erase(it);
          } else {
            (*it)->accept(*this);
            ++it;
          }
        }
      }
    };
    ControlFlowDescriptor& controlFlow = instantiation_->getIIR()->getControlFlowDescriptor();
    std::set<int> emptyStencilIDsRemoved;
    // Remove empty stencils
    for(auto it = instantiation_->getIIR()->childrenBegin();
        it != instantiation_->getIIR()->childrenEnd();) {
      Stencil& stencil = **it;
      if(stencil.isEmpty()) {
        emptyStencilIDsRemoved.insert(stencil.getStencilID());
        it = instantiation_->getIIR()->childrenErase(it);
      } else
        ++it;
    }

    controlFlow.removeStencilCalls(emptyStencilIDsRemoved, metadata_);

    // Remove the nested VerticalRegionDeclStmts and StencilCallDeclStmts
    RemoveStencilDescNodes remover;
    for(auto& statement : scope_.top()->controlFlowDescriptor_.getStatements())
      statement->ASTStmt->accept(remover);
  }

  /// @brief Push back a new statement to the end of the current statement list
  void pushBackStatement(const std::shared_ptr<iir::Stmt>& stmt) {
    scope_.top()->controlFlowDescriptor_.insertStmt(
        std::make_shared<Statement>(stmt, scope_.top()->StackTrace));
  }

  void visit(const std::shared_ptr<iir::BlockStmt>& stmt) override {
    scope_.top()->ScopeDepth++;
    for(const auto& s : stmt->getStatements()) {
      s->accept(*this);
    }
    scope_.top()->ScopeDepth--;
  }

  void visit(const std::shared_ptr<iir::ExprStmt>& stmt) override {
    if(scope_.top()->ScopeDepth == 1)
      pushBackStatement(stmt);
    stmt->getExpr()->accept(*this);
  }

  void visit(const std::shared_ptr<iir::ReturnStmt>&) override {
    DAWN_ASSERT_MSG(0, "ReturnStmt not allowed in this context");
  }

  void visit(const std::shared_ptr<iir::IfStmt>& stmt) override {
    bool result;
    if(iir::evalExprAsBoolean(stmt->getCondExpr(), result, scope_.top()->VariableMap)) {

      if(scope_.top()->ScopeDepth == 1) {
        // The condition is known at compile time, we can remove this If statement completely by
        // just not inserting it into the statement list
        if(result) {
          BlockStmt* thenBody = dyn_cast<iir::BlockStmt>(stmt->getThenStmt().get());
          DAWN_ASSERT_MSG(thenBody, "then-body of if-statment should be a BlockStmt!");
          for(auto& s : thenBody->getStatements())
            s->accept(*this);
        } else if(stmt->hasElse()) {
          BlockStmt* elseBody = dyn_cast<iir::BlockStmt>(stmt->getElseStmt().get());
          DAWN_ASSERT_MSG(elseBody, "else-body of if-statment should be a BlockStmt!");
          for(auto& s : elseBody->getStatements())
            s->accept(*this);
        }
      } else {
        // We are inside a nested statement and we need to remove this if-statment and replace it
        // with either the then-block or the else-block or in case we evaluted to `false` and
        // there
        // is no else-block we insert a `0` void statement.

        if(result) {
          // Replace the if-statement with the then-block
          // TODO very repetitive scope_.top()->control....getStatements() ...
          iir::replaceOldStmtWithNewStmtInStmt(
              scope_.top()->controlFlowDescriptor_.getStatements().back()->ASTStmt, stmt,
              stmt->getThenStmt());
          stmt->getThenStmt()->accept(*this);
        } else if(stmt->hasElse()) {
          // Replace the if-statement with the else-block
          iir::replaceOldStmtWithNewStmtInStmt(
              scope_.top()->controlFlowDescriptor_.getStatements().back()->ASTStmt, stmt,
              stmt->getElseStmt());
          stmt->getElseStmt()->accept(*this);
        } else {
          // Replace the if-statement with a void `0`
          auto voidExpr = std::make_shared<iir::LiteralAccessExpr>("0", BuiltinTypeID::Float);
          auto voidStmt = std::make_shared<iir::ExprStmt>(voidExpr);
          int AccessID = -instantiation_->nextUID();
          metadata_.insertAccessOfType(iir::FieldAccessType::FAT_Literal, AccessID, "0");
          metadata_.insertExprToAccessID(voidExpr, AccessID);
          iir::replaceOldStmtWithNewStmtInStmt(
              scope_.top()->controlFlowDescriptor_.getStatements().back()->ASTStmt, stmt, voidStmt);
        }
      }

    } else {
      if(scope_.top()->ScopeDepth == 1)
        pushBackStatement(stmt);

      stmt->getCondExpr()->accept(*this);

      // The then-part needs to go into a separate stencil ...
      makeNewStencil();
      stmt->getThenStmt()->accept(*this);

      if(stmt->hasElse()) {
        // ... same for the else-part
        makeNewStencil();
        stmt->getElseStmt()->accept(*this);
      }

      // Everything that follows needs to go into a new stencil as well
      makeNewStencil();
    }
  }

  void visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) override {
    // This is the first time we encounter this variable. We have to make sure the name is not
    // already used in another scope!

    int AccessID = metadata_.insertStmt(
        instantiation_->getOptimizerContext()->getOptions().KeepVarnames, stmt);

    // Add the mapping to the local scope
    scope_.top()->LocalVarNameToAccessIDMap.emplace(stmt->getName(), AccessID);

    // Push back the statement and move on
    if(scope_.top()->ScopeDepth == 1)
      pushBackStatement(stmt);

    // Resolve the RHS
    for(const auto& expr : stmt->getInitList())
      expr->accept(*this);

    // Check if we can evaluate the RHS to a constant expression
    if(stmt->getInitList().size() == 1) {
      double result;
      if(iir::evalExprAsDouble(stmt->getInitList().front(), result, scope_.top()->VariableMap))
        scope_.top()->VariableMap[stmt->getName()] = result;
    }
  }

  void visit(const std::shared_ptr<iir::VerticalRegionDeclStmt>& stmt) override {
    sir::VerticalRegion* verticalRegion = stmt->getVerticalRegion().get();

    tryReplaceStencilDescStmt(stmt);

    Interval interval(*verticalRegion->VerticalInterval);

    // Note that we may need to operate on copies of the ASTs because we want to have a *unique*
    // mapping of AST nodes to AccessIDs, hence we clone the ASTs of the vertical regions of
    // stencil calls
    bool cloneAST = scope_.size() > 1;
    std::shared_ptr<iir::AST> ast = cloneAST ? verticalRegion->Ast->clone() : verticalRegion->Ast;

    // Create the new multi-stage
    std::unique_ptr<MultiStage> multiStage = make_unique<MultiStage>(
        metadata_, verticalRegion->LoopOrder == sir::VerticalRegion::LK_Forward
                       ? LoopOrderKind::LK_Forward
                       : LoopOrderKind::LK_Backward);
    std::unique_ptr<Stage> stage =
        make_unique<Stage>(metadata_, instantiation_->nextUID(), interval);

    DAWN_LOG(INFO) << "Processing vertical region at " << verticalRegion->Loc;

    // Here we convert the AST of the vertical region to a flat list of statements of the stage.
    // Further, we instantiate all referenced stencil functions.
    DAWN_LOG(INFO) << "Inserting statements ... ";
    DoMethod& doMethod = stage->getSingleDoMethod();
    // TODO move iterators of IIRNode to const getChildren, when we pass here begin, end instead

    StatementMapper statementMapper(sir_, instantiation_.get(), scope_.top()->StackTrace, doMethod,
                                    doMethod.getInterval(),
                                    scope_.top()->LocalFieldnameToAccessIDMap, nullptr);
    ast->accept(statementMapper);
    DAWN_LOG(INFO) << "Inserted " << doMethod.getChildren().size() << " statements";

    if(instantiation_->getOptimizerContext()->getDiagnostics().hasErrors())
      return;
    // Here we compute the *actual* access of each statement and associate access to the AccessIDs
    // we set previously.
    DAWN_LOG(INFO) << "Filling accesses ...";
    computeAccesses(instantiation_.get(), doMethod.getChildren());

    // Now, we compute the fields of each stage (this will give us the IO-Policy of the fields)
    stage->update(iir::NodeUpdateType::level);

    // Put the stage into a separate MultiStage ...
    multiStage->insertChild(std::move(stage));

    // ... and append the MultiStages of the current stencil
    const auto& stencil = instantiation_->getIIR()->getChildren().back();
    stencil->insertChild(std::move(multiStage));
  }

  void visit(const std::shared_ptr<iir::StencilCallDeclStmt>& stmt) override {
    sir::StencilCall* stencilCall = stmt->getStencilCall().get();

    tryReplaceStencilDescStmt(stmt);

    DAWN_LOG(INFO) << "Processing stencil call to `" << stencilCall->Callee << "` at "
                   << stencilCall->Loc;

    // Prepare a new scope for the stencil call
    std::shared_ptr<Scope>& curScope = scope_.top();
    std::shared_ptr<Scope> candiateScope =
        std::make_shared<Scope>(curScope->Name, curScope->controlFlowDescriptor_);

    // Variables are inherited from the parent scope (note that this *needs* to be a copy as we
    // cannot modify the parent scope)
    candiateScope->VariableMap = curScope->VariableMap;

    // Record the call
    if(!curScope->StackTrace)
      candiateScope->StackTrace = std::make_shared<std::vector<sir::StencilCall*>>();
    else
      candiateScope->StackTrace =
          std::make_shared<std::vector<sir::StencilCall*>>(*curScope->StackTrace);
    candiateScope->StackTrace->push_back(stencilCall);

    // Get the sir::Stencil from the callee name
    auto stencilIt = std::find_if(
        sir_->Stencils.begin(), sir_->Stencils.end(),
        [&](const std::shared_ptr<sir::Stencil>& s) { return s->Name == stencilCall->Callee; });
    DAWN_ASSERT(stencilIt != sir_->Stencils.end());
    sir::Stencil& stencil = **stencilIt;

    // We need less or an equal amount of args as temporaries are added implicitly
    DAWN_ASSERT(stencilCall->Args.size() <= stencil.Fields.size());

    // Map the field arguments
    for(std::size_t stencilArgIdx = 0, stencilCallArgIdx = 0; stencilArgIdx < stencil.Fields.size();
        ++stencilArgIdx) {

      int AccessID = 0;
      if(stencil.Fields[stencilArgIdx]->IsTemporary) {
        // We add a new temporary field for each temporary field argument
        AccessID = metadata_.insertTmpField(iir::FieldAccessType::FAT_StencilTemporary,
                                            stencil.Fields[stencilArgIdx]->Name, {1, 1, 1});
      } else {
        AccessID =
            curScope->LocalFieldnameToAccessIDMap.at(stencilCall->Args[stencilCallArgIdx]->Name);
        stencilCallArgIdx++;
      }

      candiateScope->LocalFieldnameToAccessIDMap.emplace(stencil.Fields[stencilArgIdx]->Name,
                                                         AccessID);
    }

    // Process the stencil description AST of the callee.
    scope_.push(candiateScope);

    // As we *may* modify the AST we better make a copy here otherwise we get funny surprises if
    // we call this stencil multiple times ...
    stencil.StencilDescAst->clone()->accept(*this);

    scope_.pop();

    DAWN_LOG(INFO) << "Done processing stencil call to `" << stencilCall->Callee << "`";
  }

  void visit(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt) override {
    if(instantiation_->insertBoundaryConditions(stmt->getFields()[0]->Name, stmt) == false)
      DAWN_ASSERT_MSG(false, "Boundary Condition specified twice for the same field");
    //      if(instantiation_->insertBoundaryConditions(stmt->getFields()[0]->Name, stmt) ==
    //      false) DAWN_ASSERT_MSG(false, "Boundary Condition specified twice for the same
    //      field");
  }

  void visit(const std::shared_ptr<iir::AssignmentExpr>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);

    // If the LHS is known to be a known constant, we need to update its value or remove it as
    // being compile time constant
    if(VarAccessExpr* var = dyn_cast<iir::VarAccessExpr>(expr->getLeft().get())) {
      if(scope_.top()->VariableMap.count(var->getName())) {
        double result;
        if(iir::evalExprAsDouble(expr->getRight(), result, scope_.top()->VariableMap)) {
          if(StringRef(expr->getOp()) == "=")
            scope_.top()->VariableMap[var->getName()] = result;
          else if(StringRef(expr->getOp()) == "+=")
            scope_.top()->VariableMap[var->getName()] += result;
          else if(StringRef(expr->getOp()) == "-=")
            scope_.top()->VariableMap[var->getName()] -= result;
          else if(StringRef(expr->getOp()) == "*=")
            scope_.top()->VariableMap[var->getName()] *= result;
          else if(StringRef(expr->getOp()) == "/=")
            scope_.top()->VariableMap[var->getName()] /= result;
          else // unknown operator
            scope_.top()->VariableMap.erase(var->getName());
        } else
          scope_.top()->VariableMap.erase(var->getName());
      }
    }
  }

  void visit(const std::shared_ptr<iir::UnaryOperator>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
  }

  void visit(const std::shared_ptr<iir::BinaryOperator>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
  }

  void visit(const std::shared_ptr<iir::TernaryOperator>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
  }

  void visit(const std::shared_ptr<iir::FunCallExpr>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
  }

  void visit(const std::shared_ptr<iir::StencilFunCallExpr>&) override {
    DAWN_ASSERT_MSG(0, "StencilFunCallExpr not allowed in this context");
  }
  void visit(const std::shared_ptr<iir::StencilFunArgExpr>&) override {
    DAWN_ASSERT_MSG(0, "StencilFunArgExpr not allowed in this context");
  }

  void visit(const std::shared_ptr<iir::VarAccessExpr>& expr) override {
    const auto& varname = expr->getName();
    if(expr->isExternal()) {
      DAWN_ASSERT_MSG(!expr->isArrayAccess(), "global array access is not supported");

      const auto& value = instantiation_->getGlobalVariableValue(varname);
      if(value.isConstexpr()) {
        // Replace the variable access with the actual value
        DAWN_ASSERT_MSG(!value.empty(), "constant global variable with no value");

        auto newExpr = std::make_shared<dawn::LiteralAccessExpr>(
            value.toString(), sir::Value::typeToBuiltinTypeID(value.getType()));
        iir::replaceOldExprWithNewExprInStmt(
            scope_.top()->controlFlowDescriptor_.getStatements().back()->ASTStmt, expr, newExpr);

        int AccessID = instantiation_->nextUID();
        metadata_.insertAccessOfType(iir::FieldAccessType::FAT_Literal, AccessID,
                                     newExpr->getValue());
        metadata_.insertExprToAccessID(newExpr, AccessID);

      } else {
        metadata_.insertExprToAccessID(expr, metadata_.getAccessIDFromName(varname));
      }

    } else {
      // Register the mapping between VarAccessExpr and AccessID.
      metadata_.insertExprToAccessID(expr, scope_.top()->LocalVarNameToAccessIDMap[varname]);

      // Resolve the index if this is an array access
      if(expr->isArrayAccess())
        expr->getIndex()->accept(*this);
    }
  }

  void visit(const std::shared_ptr<iir::LiteralAccessExpr>& expr) override {
    // Register a literal access (Note: the negative AccessID we assign!)
    int AccessID = -instantiation_->nextUID();
    metadata_.insertAccessOfType(iir::FieldAccessType::FAT_Literal, AccessID, expr->getValue());
    metadata_.insertExprToAccessID(expr, AccessID);
  }
  void visit(const std::shared_ptr<iir::ReductionOverNeighborStmt>& stmt) override {
    pushBackStatement(stmt);
  }

  void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override {}
};
} // namespace

OptimizerContext::OptimizerContext(DiagnosticsEngine& diagnostics, Options& options,
                                   const std::shared_ptr<SIR>& SIR)
    : diagnostics_(diagnostics), options_(options), SIR_(SIR) {
  DAWN_LOG(INFO) << "Intializing OptimizerContext ... ";

  /// Instead of getting the IIR from the SIR we're generating it here:
  stencilInstantiationMap_.insert(
      std::make_pair("<unstructured>", std::make_shared<iir::StencilInstantiation>(this)));
  createIIRInMemory(stencilInstantiationMap_.at("<unstructured>"));
  if(options.Debug) {
    stencilInstantiationMap_.at("<unstructured>")->dump();
  }

  // for(const auto& stencil : SIR_->Stencils)
  //   if(!stencil->Attributes.has(sir::Attr::AK_NoCodeGen)) {
  //     stencilInstantiationMap_.insert(
  //         std::make_pair(stencil->Name, std::make_shared<iir::StencilInstantiation>(this)));
  //     fillIIRFromSIR(stencilInstantiationMap_.at(stencil->Name), stencil, SIR_);
  //   } else {
  //     DAWN_LOG(INFO) << "Skipping processing of `" << stencil->Name << "`";
  //   }
}

bool OptimizerContext::fillIIRFromSIR(
    std::shared_ptr<iir::StencilInstantiation> stencilInstantation,
    const std::shared_ptr<sir::Stencil> SIRStencil, const std::shared_ptr<SIR> fullSIR) {
  DAWN_LOG(INFO) << "Intializing StencilInstantiation of `" << SIRStencil->Name << "`";
  DAWN_ASSERT_MSG(SIRStencil, "Stencil does not exist");
  auto& metadata = stencilInstantation->getMetaData();
  metadata.setStencilname(SIRStencil->Name);
  metadata.setFileName(fullSIR->Filename);
  metadata.setStencilLocation(SIRStencil->Loc);

  // Map the fields of the "main stencil" to unique IDs (which are used in the access maps to
  // indentify the field).
  for(const auto& field : SIRStencil->Fields) {
    metadata.insertField((field->IsTemporary ? iir::FieldAccessType::FAT_StencilTemporary
                                             : iir::FieldAccessType::FAT_APIField),
                         field->Name, field->fieldDimensions);
  }

  StencilDescStatementMapper stencilDeclMapper(stencilInstantation, SIRStencil.get(), fullSIR);

  //  // We need to operate on a copy of the AST as we may modify the nodes inplace
  auto AST = SIRStencil->StencilDescAst->clone();
  AST->accept(stencilDeclMapper);

  //  Cleanup the `stencilDescStatements` and remove the empty stencils which may have been inserted
  stencilDeclMapper.cleanupStencilDeclAST();

  //  // Repair broken references to temporaries i.e promote them to real fields
  PassTemporaryType::fixTemporariesSpanningMultipleStencils(
      stencilInstantation.get(), stencilInstantation->getIIR()->getChildren());

  if(stencilInstantation->getOptimizerContext()->getOptions().ReportAccesses) {
    stencilInstantation->reportAccesses();
  }

  for(const auto& MS : iterateIIROver<MultiStage>(*(stencilInstantation->getIIR()))) {
    MS->update(NodeUpdateType::levelAndTreeAbove);
  }
  DAWN_LOG(INFO) << "Done initializing StencilInstantiation";

  // Iterate all statements (top -> bottom)
  for(const auto& stagePtr : iterateIIROver<iir::Stage>(*(stencilInstantation->getIIR()))) {
    iir::Stage& stage = *stagePtr;
    for(const auto& doMethod : stage.getChildren()) {
      doMethod->update(iir::NodeUpdateType::level);
    }
    stage.update(iir::NodeUpdateType::level);
  }
  for(const auto& MSPtr : iterateIIROver<iir::Stage>(*(stencilInstantation->getIIR()))) {
    MSPtr->update(iir::NodeUpdateType::levelAndTreeAbove);
  }

  return true;
}

std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
OptimizerContext::getStencilInstantiationMap() {
  return stencilInstantiationMap_;
}

const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
OptimizerContext::getStencilInstantiationMap() const {
  return stencilInstantiationMap_;
}

const DiagnosticsEngine& OptimizerContext::getDiagnostics() const { return diagnostics_; }

DiagnosticsEngine& OptimizerContext::getDiagnostics() { return diagnostics_; }

const Options& OptimizerContext::getOptions() const { return options_; }

Options& OptimizerContext::getOptions() { return options_; }

} // namespace dawn
