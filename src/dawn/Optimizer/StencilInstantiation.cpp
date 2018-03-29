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

#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/Optimizer/AccessComputation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassTemporaryType.h"
#include "dawn/Optimizer/Renaming.h"
#include "dawn/Optimizer/Replacing.h"
#include "dawn/Optimizer/StatementAccessesPair.h"
#include "dawn/Optimizer/StatementMapper.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTUtil.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/FileUtil.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Json.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/Printing.h"
#include "dawn/Support/Twine.h"
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <stack>

namespace dawn {

namespace {

//===------------------------------------------------------------------------------------------===//
//     StencilDescStatementMapper
//===------------------------------------------------------------------------------------------===//

/// @brief Map the statements of the stencil description AST to a flat list of statements and inline
/// all calls to other stencils
class StencilDescStatementMapper : public ASTVisitor {

  /// @brief Record of the current scope (each StencilCall will create a new scope)
  struct Scope : public NonCopyable {
    Scope(const std::string& name, std::vector<std::shared_ptr<Statement>>& statements)
        : Name(name), ScopeDepth(0), Statements(statements), StackTrace(nullptr) {}

    /// Name of the current stencil
    std::string Name;

    /// Nesting of scopes
    int ScopeDepth;

    /// List of statements of the stencil description
    std::vector<std::shared_ptr<Statement>>& Statements;

    /// Scope fieldnames to to (global) AccessID
    std::unordered_map<std::string, int> LocalFieldnameToAccessIDMap;

    /// Scope variable name to (global) AccessID
    std::unordered_map<std::string, int> LocalVarNameToAccessIDMap;

    /// Map of known values of variables
    std::unordered_map<std::string, double> VariableMap;

    /// Current call stack of stencil calls (may be NULL)
    std::shared_ptr<std::vector<sir::StencilCall*>> StackTrace;
  };

  StencilInstantiation* instantiation_;
  std::stack<std::shared_ptr<Scope>> scope_;

  /// We replace the first VerticalRegionDeclStmt with a dummy node which signals code-gen that it
  /// should insert a call to the gridtools stencil here
  std::shared_ptr<Stmt> stencilDescReplacement_;

public:
  StencilDescStatementMapper(StencilInstantiation* instantiation, const std::string& name,
                             std::vector<std::shared_ptr<Statement>>& statements,
                             const std::unordered_map<std::string, int>& fieldnameToAccessIDMap)
      : instantiation_(instantiation) {
    DAWN_ASSERT(instantiation);
    // Create the initial scope
    scope_.push(std::make_shared<Scope>(name, statements));
    scope_.top()->LocalFieldnameToAccessIDMap = fieldnameToAccessIDMap;

    // We add all global variables which have constant values
    for(const auto& keyValuePair : *instantiation->getSIR()->GlobalVariableMap) {
      const std::string& key = keyValuePair.first;
      const sir::Value& value = *keyValuePair.second;

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

  /// @brief Create a new stencil in the instantiation and prepare the replacement node for the next
  /// VerticalRegionDeclStmt
  ///
  /// @see tryReplaceVerticalRegionDeclStmt
  void makeNewStencil() {
    int StencilID = instantiation_->nextUID();
    instantiation_->getStencils().emplace_back(
        std::make_shared<Stencil>(*instantiation_, instantiation_->getSIRStencil(), StencilID));

    // We create a paceholder stencil-call for CodeGen to know wehere we need to insert calls to
    // this stencil
    auto placeholderStencil = std::make_shared<sir::StencilCall>(
        StencilInstantiation::makeStencilCallCodeGenName(StencilID));
    auto stencilCallDeclStmt = std::make_shared<StencilCallDeclStmt>(placeholderStencil);

    // Register the call and set it as a replacement for the next vertical region
    instantiation_->getStencilCallToStencilIDMap().emplace(stencilCallDeclStmt, StencilID);
    stencilDescReplacement_ = stencilCallDeclStmt;
  }

  /// @brief Replace the first VerticalRegionDeclStmt or StencilCallDelcStmt with a dummy
  /// placeholder signaling code-gen that it should insert a call to the gridtools stencil.
  ///
  /// All remaining VerticalRegion/StencilCalls statements which are still in the stencil
  /// description AST are pruned at the end
  ///
  /// @see removeObsoleteStencilDescNodes
  void tryReplaceStencilDescStmt(const std::shared_ptr<Stmt>& stencilDescNode) {
    DAWN_ASSERT(stencilDescNode->isStencilDesc());

    // Nothing to do, statement was already replaced
    if(!stencilDescReplacement_)
      return;

    // Instead of inserting the VerticalRegionDeclStmt we insert the call to the gridtools stencil
    if(scope_.top()->ScopeDepth == 1)
      scope_.top()->Statements.emplace_back(
          std::make_shared<Statement>(stencilDescReplacement_, scope_.top()->StackTrace));
    else {

      // We need to replace the VerticalRegionDeclStmt in the current statement
      replaceOldStmtWithNewStmtInStmt(scope_.top()->Statements.back()->ASTStmt, stencilDescNode,
                                      stencilDescReplacement_);
    }

    stencilDescReplacement_ = nullptr;
  }

  /// @brief Remove all VerticalRegionDeclStmt and StencilCallDeclStmt (which do not start with
  /// `GridToolsStencilCallPrefix`) from the list of statements and remove empty stencils
  void cleanupStencilDeclAST() {

    // We only need to remove "nested" nodes as the top-level VerticalRegions or StencilCalls are
    // not inserted into the statement list in the frist place
    class RemoveStencilDescNodes : public ASTVisitorForwarding {
    public:
      bool needsRemoval(const std::shared_ptr<Stmt>& stmt) const {
        if(StencilCallDeclStmt* s = dyn_cast<StencilCallDeclStmt>(stmt.get())) {
          // StencilCallDeclStmt node, remove it if it is not one of our artificial stencil call
          // nodes
          if(!StencilInstantiation::isStencilCallCodeGenName(s->getStencilCall()->Callee))
            return true;

        } else if(isa<VerticalRegionDeclStmt>(stmt.get()))
          // Remove all remaining vertical regions
          return true;

        return false;
      }

      void visit(const std::shared_ptr<BlockStmt>& stmt) override {
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

    // Remove the nested VerticalRegionDeclStmts and StencilCallDeclStmts
    RemoveStencilDescNodes remover;
    for(auto& statement : scope_.top()->Statements)
      statement->ASTStmt->accept(remover);

    // Remove empty stencils
    for(auto it = instantiation_->getStencils().begin();
        it != instantiation_->getStencils().end();) {
      Stencil& stencil = **it;
      if(stencil.isEmpty())
        it = instantiation_->getStencils().erase(it);
      else
        ++it;
    }
  }

  /// @brief Push back a new statement to the end of the current statement list
  void pushBackStatement(const std::shared_ptr<Stmt>& stmt) {
    scope_.top()->Statements.emplace_back(
        std::make_shared<Statement>(stmt, scope_.top()->StackTrace));
  }

  void visit(const std::shared_ptr<BlockStmt>& stmt) override {
    scope_.top()->ScopeDepth++;
    for(const auto& s : stmt->getStatements()) {
      s->accept(*this);
    }
    scope_.top()->ScopeDepth--;
  }

  void visit(const std::shared_ptr<ExprStmt>& stmt) override {
    if(scope_.top()->ScopeDepth == 1)
      pushBackStatement(stmt);
    stmt->getExpr()->accept(*this);
  }

  void visit(const std::shared_ptr<ReturnStmt>& stmt) override {
    DAWN_ASSERT_MSG(0, "ReturnStmt not allowed in this context");
  }

  void visit(const std::shared_ptr<IfStmt>& stmt) override {
    bool result;
    if(evalExprAsBoolean(stmt->getCondExpr(), result, scope_.top()->VariableMap)) {

      if(scope_.top()->ScopeDepth == 1) {
        // The condition is known at compile time, we can remove this If statement completely by
        // just not inserting it into the statement list
        if(result) {
          BlockStmt* thenBody = dyn_cast<BlockStmt>(stmt->getThenStmt().get());
          DAWN_ASSERT_MSG(thenBody, "then-body of if-statment should be a BlockStmt!");
          for(auto& s : thenBody->getStatements())
            s->accept(*this);
        } else if(stmt->hasElse()) {
          BlockStmt* elseBody = dyn_cast<BlockStmt>(stmt->getElseStmt().get());
          DAWN_ASSERT_MSG(elseBody, "else-body of if-statment should be a BlockStmt!");
          for(auto& s : elseBody->getStatements())
            s->accept(*this);
        }
      } else {
        // We are inside a nested statement and we need to remove this if-statment and replace it
        // with either the then-block or the else-block or in case we evaluted to `false` and there
        // is no else-block we insert a `0` void statement.

        if(result) {
          // Replace the if-statement with the then-block
          replaceOldStmtWithNewStmtInStmt(scope_.top()->Statements.back()->ASTStmt, stmt,
                                          stmt->getThenStmt());
          stmt->getThenStmt()->accept(*this);
        } else if(stmt->hasElse()) {
          // Replace the if-statement with the else-block
          replaceOldStmtWithNewStmtInStmt(scope_.top()->Statements.back()->ASTStmt, stmt,
                                          stmt->getElseStmt());
          stmt->getElseStmt()->accept(*this);
        } else {
          // Replace the if-statement with a void `0`
          auto voidExpr = std::make_shared<LiteralAccessExpr>("0", BuiltinTypeID::Float);
          auto voidStmt = std::make_shared<ExprStmt>(voidExpr);
          int AccessID = -instantiation_->nextUID();
          instantiation_->getLiteralAccessIDToNameMap().emplace(AccessID, "0");
          instantiation_->mapExprToAccessID(voidExpr, AccessID);
          replaceOldStmtWithNewStmtInStmt(scope_.top()->Statements.back()->ASTStmt, stmt, voidStmt);
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

  void visit(const std::shared_ptr<VarDeclStmt>& stmt) override {
    // This is the first time we encounter this variable. We have to make sure the name is not
    // already used in another scope!
    int AccessID = instantiation_->nextUID();

    std::string globalName;
    if(instantiation_->getOptimizerContext()->getOptions().KeepVarnames)
      globalName = stmt->getName();
    else
      globalName = StencilInstantiation::makeLocalVariablename(stmt->getName(), AccessID);

    instantiation_->setAccessIDNamePair(AccessID, globalName);
    instantiation_->getStmtToAccessIDMap().emplace(stmt, AccessID);

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
      if(evalExprAsDouble(stmt->getInitList().front(), result, scope_.top()->VariableMap))
        scope_.top()->VariableMap[stmt->getName()] = result;
    }
  }

  void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) override {
    sir::VerticalRegion* verticalRegion = stmt->getVerticalRegion().get();

    tryReplaceStencilDescStmt(stmt);

    Interval interval(*verticalRegion->VerticalInterval);

    // Note that we may need to operate on copies of the ASTs because we want to have a *unique*
    // mapping of AST nodes to AccessIDs, hence we clone the ASTs of the vertical regions of
    // stencil calls
    bool cloneAST = scope_.size() > 1;
    std::shared_ptr<AST> ast = cloneAST ? verticalRegion->Ast->clone() : verticalRegion->Ast;

    // Create the new multi-stage
    std::shared_ptr<MultiStage> multiStage = std::make_shared<MultiStage>(
        *instantiation_, verticalRegion->LoopOrder == sir::VerticalRegion::LK_Forward
                             ? LoopOrderKind::LK_Forward
                             : LoopOrderKind::LK_Backward);
    std::shared_ptr<Stage> stage = std::make_shared<Stage>(*instantiation_, multiStage.get(),
                                                           instantiation_->nextUID(), interval);

    DAWN_LOG(INFO) << "Processing vertical region at " << verticalRegion->Loc;

    // Here we convert the AST of the vertical region to a flat list of statements of the stage.
    // Further, we instantiate all referenced stencil functions.
    DAWN_LOG(INFO) << "Inserting statements ... ";
    DoMethod& doMethod = stage->getSingleDoMethod();
    StatementMapper statementMapper(instantiation_, scope_.top()->StackTrace,
                                    doMethod.getStatementAccessesPairs(), doMethod.getInterval(),
                                    scope_.top()->LocalFieldnameToAccessIDMap, nullptr);
    ast->accept(statementMapper);
    DAWN_LOG(INFO) << "Inserted " << doMethod.getStatementAccessesPairs().size() << " statements";

    if(instantiation_->getOptimizerContext()->getDiagnostics().hasErrors())
      return;

    // Here we compute the *actual* access of each statement and associate access to the AccessIDs
    // we set previously.
    DAWN_LOG(INFO) << "Filling accesses ...";
    computeAccesses(instantiation_, doMethod.getStatementAccessesPairs());

    // Now, we compute the fields of each stage (this will give us the IO-Policy of the fields)
    stage->update();

    // Put the stage into a separate MultiStage ...
    multiStage->getStages().push_back(std::move(stage));

    // ... and append the MultiStages of the current stencil
    instantiation_->getStencils().back()->getMultiStages().push_back(std::move(multiStage));
  }

  void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override {
    sir::StencilCall* stencilCall = stmt->getStencilCall().get();

    tryReplaceStencilDescStmt(stmt);

    DAWN_LOG(INFO) << "Processing stencil call to `" << stencilCall->Callee << "` at "
                   << stencilCall->Loc;

    // Prepare a new scope for the stencil call
    std::shared_ptr<Scope>& curScope = scope_.top();
    std::shared_ptr<Scope> candiateScope =
        std::make_shared<Scope>(curScope->Name, curScope->Statements);

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
        instantiation_->getSIR()->Stencils.begin(), instantiation_->getSIR()->Stencils.end(),
        [&](const std::shared_ptr<sir::Stencil>& s) { return s->Name == stencilCall->Callee; });
    DAWN_ASSERT(stencilIt != instantiation_->getSIR()->Stencils.end());
    sir::Stencil& stencil = **stencilIt;

    // We need less or an equal amount of args as temporaries are added implicitly
    DAWN_ASSERT(stencilCall->Args.size() <= stencil.Fields.size());

    // Map the field arguments
    for(std::size_t stencilArgIdx = 0, stencilCallArgIdx = 0; stencilArgIdx < stencil.Fields.size();
        ++stencilArgIdx) {

      int AccessID = 0;
      if(stencil.Fields[stencilArgIdx]->IsTemporary) {
        // We add a new temporary field for each temporary field argument
        AccessID = instantiation_->nextUID();
        instantiation_->setAccessIDNamePairOfField(
            AccessID, StencilInstantiation::makeTemporaryFieldname(
                          stencil.Fields[stencilArgIdx]->Name, AccessID),
            true);
      } else {
        AccessID =
            curScope->LocalFieldnameToAccessIDMap.find(stencilCall->Args[stencilCallArgIdx]->Name)
                ->second;
        stencilCallArgIdx++;
      }

      candiateScope->LocalFieldnameToAccessIDMap.emplace(stencil.Fields[stencilArgIdx]->Name,
                                                         AccessID);
    }

    // Process the stencil description AST of the callee.
    scope_.push(candiateScope);

    // As we *may* modify the AST we better make a copy here otherwise we get funny surprises if we
    // call this stencil multiple times ...
    stencil.StencilDescAst->clone()->accept(*this);

    scope_.pop();

    DAWN_LOG(INFO) << "Done processing stencil call to `" << stencilCall->Callee << "`";
  }

  void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) override {
    if(instantiation_->insertBoundaryConditions(stmt->getFields()[0]->Name, stmt) == false)
      DAWN_ASSERT_MSG(false, "Boundary Condition specified twice for the same field");
  }

  void visit(const std::shared_ptr<AssignmentExpr>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);

    // If the LHS is known to be a known constant, we need to update its value or remove it as
    // being compile time constant
    if(VarAccessExpr* var = dyn_cast<VarAccessExpr>(expr->getLeft().get())) {
      if(scope_.top()->VariableMap.count(var->getName())) {
        double result;
        if(evalExprAsDouble(expr->getRight(), result, scope_.top()->VariableMap)) {
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

  void visit(const std::shared_ptr<UnaryOperator>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
  }

  void visit(const std::shared_ptr<BinaryOperator>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
  }

  void visit(const std::shared_ptr<TernaryOperator>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
  }

  void visit(const std::shared_ptr<FunCallExpr>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
  }

  void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override {
    DAWN_ASSERT_MSG(0, "StencilFunCallExpr not allowed in this context");
  }
  void visit(const std::shared_ptr<StencilFunArgExpr>& expr) override {
    DAWN_ASSERT_MSG(0, "StencilFunArgExpr not allowed in this context");
  }

  void visit(const std::shared_ptr<VarAccessExpr>& expr) override {
    const auto& varname = expr->getName();

    if(expr->isExternal()) {
      DAWN_ASSERT_MSG(!expr->isArrayAccess(), "global array access is not supported");

      const auto& value = instantiation_->getGlobalVariableValue(varname);
      if(value.isConstexpr()) {
        // Replace the variable access with the actual value
        DAWN_ASSERT_MSG(!value.empty(), "constant global variable with no value");

        auto newExpr = std::make_shared<dawn::LiteralAccessExpr>(
            value.toString(), sir::Value::typeToBuiltinTypeID(value.getType()));
        replaceOldExprWithNewExprInStmt(scope_.top()->Statements.back()->ASTStmt, expr, newExpr);

        int AccessID = instantiation_->nextUID();
        instantiation_->getLiteralAccessIDToNameMap().emplace(AccessID, newExpr->getValue());
        instantiation_->mapExprToAccessID(newExpr, AccessID);

      } else {
        int AccessID = 0;
        if(!instantiation_->isGlobalVariable(varname)) {
          AccessID = instantiation_->nextUID();
          instantiation_->setAccessIDNamePairOfGlobalVariable(AccessID, varname);
        } else {
          AccessID = instantiation_->getAccessIDFromName(varname);
        }

        instantiation_->mapExprToAccessID(expr, AccessID);
      }

    } else {
      // Register the mapping between VarAccessExpr and AccessID.
      instantiation_->mapExprToAccessID(expr, scope_.top()->LocalVarNameToAccessIDMap[varname]);

      // Resolve the index if this is an array access
      if(expr->isArrayAccess())
        expr->getIndex()->accept(*this);
    }
  }

  void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override {
    // Register a literal access (Note: the negative AccessID we assign!)
    int AccessID = -instantiation_->nextUID();
    instantiation_->getLiteralAccessIDToNameMap().emplace(AccessID, expr->getValue());
    instantiation_->mapExprToAccessID(expr, AccessID);
  }

  void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {}
};

} // anonymous namespace

//===------------------------------------------------------------------------------------------===//
//     StencilInstantiation
//===------------------------------------------------------------------------------------------===//

StencilInstantiation::StencilInstantiation(OptimizerContext* context,
                                           std::shared_ptr<sir::Stencil> const& SIRStencil,
                                           std::shared_ptr<SIR> const& SIR)
    : context_(context), SIRStencil_(SIRStencil), SIR_(SIR) {
  DAWN_LOG(INFO) << "Intializing StencilInstantiation of `" << SIRStencil->Name << "`";
  DAWN_ASSERT_MSG(SIRStencil, "Stencil does not exist");

  // Map the fields of the "main stencil" to unique IDs (which are used in the access maps to
  // indentify the field).
  for(const auto& field : SIRStencil->Fields) {
    int AccessID = nextUID();
    setAccessIDNamePairOfField(AccessID, field->Name, field->IsTemporary);
  }

  // Process the stencil description of the "main stencil"
  StencilDescStatementMapper stencilDeclMapper(this, getName(), stencilDescStatements_,
                                               NameToAccessIDMap_);

  // We need to operate on a copy of the AST as we may modify the nodes inplace
  auto AST = SIRStencil->StencilDescAst->clone();
  AST->accept(stencilDeclMapper);

  // Cleanup the `stencilDescStatements` and remove the empty stencils which may have been inserted
  stencilDeclMapper.cleanupStencilDeclAST();

  // Repair broken references to temporaries i.e promote them to real fields
  PassTemporaryType::fixTemporariesSpanningMultipleStencils(this, stencils_);

  if(context_->getOptions().ReportAccesses)
    reportAccesses();

  DAWN_LOG(INFO) << "Done initializing StencilInstantiation";
}

void StencilInstantiation::setAccessIDNamePair(int AccessID, const std::string& name) {
  AccessIDToNameMap_.emplace(AccessID, name);
  NameToAccessIDMap_.emplace(name, AccessID);
}

void StencilInstantiation::setAccessIDNamePairOfField(int AccessID, const std::string& name,
                                                      bool isTemporary) {
  setAccessIDNamePair(AccessID, name);
  FieldAccessIDSet_.insert(AccessID);
  if(isTemporary)
    TemporaryFieldAccessIDSet_.insert(AccessID);
}

void StencilInstantiation::setAccessIDNamePairOfGlobalVariable(int AccessID,
                                                               const std::string& name) {
  setAccessIDNamePair(AccessID, name);
  GlobalVariableAccessIDSet_.insert(AccessID);
}

void StencilInstantiation::removeAccessID(int AccessID) {
  if(NameToAccessIDMap_.count(AccessIDToNameMap_[AccessID]))
    NameToAccessIDMap_.erase(AccessIDToNameMap_[AccessID]);

  AccessIDToNameMap_.erase(AccessID);
  FieldAccessIDSet_.erase(AccessID);
  TemporaryFieldAccessIDSet_.erase(AccessID);

  if(variableVersions_.hasVariableMultipleVersions(AccessID)) {
    auto versions = variableVersions_.getVersions(AccessID);
    versions->erase(std::remove_if(versions->begin(), versions->end(),
                                   [&](int AID) { return AID == AccessID; }),
                    versions->end());
  }
}

const std::string StencilInstantiation::getName() const { return SIRStencil_->Name; }

const std::unordered_map<std::shared_ptr<Stmt>, int>&
StencilInstantiation::getStmtToAccessIDMap() const {
  return StmtToAccessIDMap_;
}

std::unordered_map<std::shared_ptr<Stmt>, int>& StencilInstantiation::getStmtToAccessIDMap() {
  return StmtToAccessIDMap_;
}

const std::string& StencilInstantiation::getNameFromAccessID(int AccessID) const {
  if(AccessID < 0)
    return getNameFromLiteralAccessID(AccessID);
  auto it = AccessIDToNameMap_.find(AccessID);
  DAWN_ASSERT_MSG(it != AccessIDToNameMap_.end(), "Invalid AccessID");
  return it->second;
}

const std::string& StencilInstantiation::getNameFromStageID(int StageID) const {
  auto it = StageIDToNameMap_.find(StageID);
  DAWN_ASSERT_MSG(it != StageIDToNameMap_.end(), "Invalid StageID");
  return it->second;
}

void StencilInstantiation::mapExprToAccessID(const std::shared_ptr<Expr>& expr, int accessID) {
  ExprToAccessIDMap_.emplace(expr, accessID);
}

void StencilInstantiation::eraseExprToAccessID(std::shared_ptr<Expr> expr) {
  DAWN_ASSERT(ExprToAccessIDMap_.count(expr));
  ExprToAccessIDMap_.erase(expr);
}

void StencilInstantiation::mapStmtToAccessID(const std::shared_ptr<Stmt>& stmt, int accessID) {
  StmtToAccessIDMap_.emplace(stmt, accessID);
}

const std::string& StencilInstantiation::getNameFromLiteralAccessID(int AccessID) const {
  DAWN_ASSERT_MSG(isLiteral(AccessID), "Invalid literal");
  return LiteralAccessIDToNameMap_.find(AccessID)->second;
}

bool StencilInstantiation::isGlobalVariable(const std::string& name) const {
  auto it = NameToAccessIDMap_.find(name);
  return it == NameToAccessIDMap_.end() ? false : isGlobalVariable(it->second);
}

void StencilInstantiation::insertStencilFunctionIntoSIR(
    const std::shared_ptr<sir::StencilFunction>& sirStencilFunction) {
  SIR_->StencilFunctions.push_back(sirStencilFunction);
}

const sir::Value& StencilInstantiation::getGlobalVariableValue(const std::string& name) const {
  auto it = getSIR()->GlobalVariableMap->find(name);
  DAWN_ASSERT(it != getSIR()->GlobalVariableMap->end());
  return *it->second;
}

ArrayRef<int> StencilInstantiation::getFieldVersions(int AccessID) const {
  return variableVersions_.hasVariableMultipleVersions(AccessID)
             ? ArrayRef<int>(*(variableVersions_.getVersions(AccessID)))
             : ArrayRef<int>{};
}

int StencilInstantiation::createVersionAndRename(int AccessID, Stencil* stencil, int curStageIdx,
                                                 int curStmtIdx, std::shared_ptr<Expr>& expr,
                                                 RenameDirection dir) {
  int newAccessID = nextUID();

  if(isField(AccessID)) {
    if(variableVersions_.hasVariableMultipleVersions(AccessID)) {
      // Field is already multi-versioned, append a new version
      auto versions = variableVersions_.getVersions(AccessID);

      // Set the second to last field to be a temporary (only the first and the last field will be
      // real storages, all other versions will be temporaries)
      int lastAccessID = versions->back();
      TemporaryFieldAccessIDSet_.insert(lastAccessID);
      AllocatedFieldAccessIDSet_.erase(lastAccessID);

      // The field with version 0 contains the original name
      const std::string& originalName = getNameFromAccessID(versions->front());

      // Register the new field
      setAccessIDNamePairOfField(newAccessID, originalName + "_" + std::to_string(versions->size()),
                                 false);
      AllocatedFieldAccessIDSet_.insert(newAccessID);

      versions->push_back(newAccessID);
      variableVersions_.insert(newAccessID, versions);

    } else {
      const std::string& originalName = getNameFromAccessID(AccessID);

      // Register the new *and* old field as being multi-versioned and indicate code-gen it has to
      // allocate the second version
      auto versionsVecPtr = std::make_shared<std::vector<int>>();
      *versionsVecPtr = {AccessID, newAccessID};

      setAccessIDNamePairOfField(newAccessID, originalName + "_1", false);
      AllocatedFieldAccessIDSet_.insert(newAccessID);

      variableVersions_.insert(AccessID, versionsVecPtr);
      variableVersions_.insert(newAccessID, versionsVecPtr);
    }
  } else {
    if(variableVersions_.hasVariableMultipleVersions(AccessID)) {
      // Variable is already multi-versioned, append a new version
      auto versions = variableVersions_.getVersions(AccessID);

      // The variable with version 0 contains the original name
      const std::string& originalName = getNameFromAccessID(versions->front());

      // Register the new variable
      setAccessIDNamePair(newAccessID, originalName + "_" + std::to_string(versions->size()));
      versions->push_back(newAccessID);
      variableVersions_.insert(newAccessID, versions);

    } else {
      const std::string& originalName = getNameFromAccessID(AccessID);

      // Register the new *and* old variable as being multi-versioned
      auto versionsVecPtr = std::make_shared<std::vector<int>>();
      *versionsVecPtr = {AccessID, newAccessID};

      setAccessIDNamePair(newAccessID, originalName + "_1");
      variableVersions_.insert(AccessID, versionsVecPtr);
      variableVersions_.insert(newAccessID, versionsVecPtr);
    }
  }

  // Rename the Expression
  renameAccessIDInExpr(this, AccessID, newAccessID, expr);

  // Recompute the accesses of the current statement (only works with single Do-Methods - for now)
  computeAccesses(
      this,
      stencil->getStage(curStageIdx)->getSingleDoMethod().getStatementAccessesPairs()[curStmtIdx]);

  // Rename the statement and accesses
  for(int stageIdx = curStageIdx;
      dir == RD_Above ? (stageIdx >= 0) : (stageIdx < stencil->getNumStages());
      dir == RD_Above ? stageIdx-- : stageIdx++) {
    Stage& stage = *stencil->getStage(stageIdx);
    DoMethod& doMethod = stage.getSingleDoMethod();

    if(stageIdx == curStageIdx) {
      for(int i = dir == RD_Above ? (curStmtIdx - 1) : (curStmtIdx + 1);
          dir == RD_Above ? (i >= 0) : (i < doMethod.getStatementAccessesPairs().size());
          dir == RD_Above ? (--i) : (++i)) {
        renameAccessIDInStmts(this, AccessID, newAccessID, doMethod.getStatementAccessesPairs()[i]);
        renameAccessIDInAccesses(this, AccessID, newAccessID,
                                 doMethod.getStatementAccessesPairs()[i]);
      }

    } else {
      renameAccessIDInStmts(this, AccessID, newAccessID, doMethod.getStatementAccessesPairs());
      renameAccessIDInAccesses(this, AccessID, newAccessID, doMethod.getStatementAccessesPairs());
    }

    // Updat the fields of the stage
    stage.update();
  }

  return newAccessID;
}

void StencilInstantiation::renameAllOccurrences(Stencil* stencil, int oldAccessID,
                                                int newAccessID) {
  // Rename the statements and accesses
  stencil->renameAllOccurrences(oldAccessID, newAccessID);

  // Remove form all AccessID maps
  removeAccessID(oldAccessID);
}

void StencilInstantiation::promoteLocalVariableToTemporaryField(Stencil* stencil, int AccessID,
                                                                const Stencil::Lifetime& lifetime) {
  std::string varname = getNameFromAccessID(AccessID);
  std::string fieldname = StencilInstantiation::makeTemporaryFieldname(
      StencilInstantiation::extractLocalVariablename(varname), AccessID);

  // Replace all variable accesses with field accesses
  stencil->forEachStatementAccessesPair(
      [&](ArrayRef<std::shared_ptr<StatementAccessesPair>> statementAccessesPair) -> void {
        replaceVarWithFieldAccessInStmts(stencil, AccessID, fieldname, statementAccessesPair);
      },
      lifetime);

  // Replace the the variable declaration with an assignment to the temporary field
  std::vector<std::shared_ptr<StatementAccessesPair>>& statementAccessesPairs =
      stencil->getStage(lifetime.Begin.StagePos)
          ->getDoMethods()[lifetime.Begin.DoMethodIndex]
          ->getStatementAccessesPairs();
  std::shared_ptr<Statement> oldStatement =
      statementAccessesPairs[lifetime.Begin.StatementIndex]->getStatement();

  // The oldStmt has to be a `VarDeclStmt`. For example
  //
  //   double __local_foo = ...
  //
  // will be replaced with
  //
  //   __tmp_foo(0, 0, 0) = ...
  //
  VarDeclStmt* varDeclStmt = dyn_cast<VarDeclStmt>(oldStatement->ASTStmt.get());
  DAWN_ASSERT_MSG(varDeclStmt,
                  "first access to variable (i.e lifetime.Begin) is not an `VarDeclStmt`");
  DAWN_ASSERT_MSG(!varDeclStmt->isArray(), "cannot promote local array to temporary field");

  auto fieldAccessExpr = std::make_shared<FieldAccessExpr>(fieldname);
  ExprToAccessIDMap_.emplace(fieldAccessExpr, AccessID);
  auto assignmentExpr =
      std::make_shared<AssignmentExpr>(fieldAccessExpr, varDeclStmt->getInitList().front());
  auto exprStmt = std::make_shared<ExprStmt>(assignmentExpr);

  // Replace the statement
  statementAccessesPairs[lifetime.Begin.StatementIndex]->setStatement(
      std::make_shared<Statement>(exprStmt, oldStatement->StackTrace));

  // Remove the variable
  removeAccessID(AccessID);
  StmtToAccessIDMap_.erase(oldStatement->ASTStmt);

  // Register the field
  setAccessIDNamePairOfField(AccessID, fieldname, true);

  // Update the fields of the stages we modified
  stencil->updateFields(lifetime);
}

void StencilInstantiation::promoteTemporaryFieldToAllocatedField(int AccessID) {
  DAWN_ASSERT(isTemporaryField(AccessID));
  TemporaryFieldAccessIDSet_.erase(AccessID);
  AllocatedFieldAccessIDSet_.insert(AccessID);
}

void StencilInstantiation::demoteTemporaryFieldToLocalVariable(Stencil* stencil, int AccessID,
                                                               const Stencil::Lifetime& lifetime) {
  std::string fieldname = getNameFromAccessID(AccessID);
  std::string varname = StencilInstantiation::makeLocalVariablename(
      StencilInstantiation::extractTemporaryFieldname(fieldname), AccessID);

  // Replace all field accesses with variable accesses
  stencil->forEachStatementAccessesPair(
      [&](ArrayRef<std::shared_ptr<StatementAccessesPair>> statementAccessesPairs) -> void {
        replaceFieldWithVarAccessInStmts(stencil, AccessID, varname, statementAccessesPairs);
      },
      lifetime);

  // Replace the first access to the field with a VarDeclStmt
  std::vector<std::shared_ptr<StatementAccessesPair>>& statementAccessesPairs =
      stencil->getStage(lifetime.Begin.StagePos)
          ->getDoMethods()[lifetime.Begin.DoMethodIndex]
          ->getStatementAccessesPairs();
  std::shared_ptr<Statement> oldStatement =
      statementAccessesPairs[lifetime.Begin.StatementIndex]->getStatement();

  // The oldStmt has to be an `ExprStmt` with an `AssignmentExpr`. For example
  //
  //   __tmp_foo(0, 0, 0) = ...
  //
  // will be replaced with
  //
  //   double __local_foo = ...
  //
  ExprStmt* exprStmt = dyn_cast<ExprStmt>(oldStatement->ASTStmt.get());
  DAWN_ASSERT_MSG(exprStmt, "first access of field (i.e lifetime.Begin) is not an `ExprStmt`");
  AssignmentExpr* assignmentExpr = dyn_cast<AssignmentExpr>(exprStmt->getExpr().get());
  DAWN_ASSERT_MSG(assignmentExpr,
                  "first access of field (i.e lifetime.Begin) is not an `AssignmentExpr`");

  // Create the new `VarDeclStmt` which will replace the old `ExprStmt`
  std::shared_ptr<Stmt> varDeclStmt =
      std::make_shared<VarDeclStmt>(Type(BuiltinTypeID::Float), varname, 0, "=",
                                    std::vector<std::shared_ptr<Expr>>{assignmentExpr->getRight()});

  // Replace the statement
  statementAccessesPairs[lifetime.Begin.StatementIndex]->setStatement(
      std::make_shared<Statement>(varDeclStmt, oldStatement->StackTrace));

  // Remove the field
  removeAccessID(AccessID);

  // Register the variable
  setAccessIDNamePair(AccessID, varname);
  StmtToAccessIDMap_.emplace(varDeclStmt, AccessID);

  // Update the fields of the stages we modified
  stencil->updateFields(lifetime);
}

int StencilInstantiation::getAccessIDFromName(const std::string& name) const {
  auto it = NameToAccessIDMap_.find(name);
  DAWN_ASSERT_MSG(it != NameToAccessIDMap_.end(), "Invalid name");
  return it->second;
}

int StencilInstantiation::getAccessIDFromExpr(const std::shared_ptr<Expr>& expr) const {
  auto it = ExprToAccessIDMap_.find(expr);
  DAWN_ASSERT_MSG(it != ExprToAccessIDMap_.end(), "Invalid Expr");
  return it->second;
}

int StencilInstantiation::getAccessIDFromStmt(const std::shared_ptr<Stmt>& stmt) const {
  auto it = StmtToAccessIDMap_.find(stmt);
  DAWN_ASSERT_MSG(it != StmtToAccessIDMap_.end(), "Invalid Stmt");
  return it->second;
}

void StencilInstantiation::setAccessIDOfStmt(const std::shared_ptr<Stmt>& stmt,
                                             const int accessID) {
  DAWN_ASSERT(StmtToAccessIDMap_.count(stmt));
  StmtToAccessIDMap_[stmt] = accessID;
}

void StencilInstantiation::setAccessIDOfExpr(const std::shared_ptr<Expr>& expr,
                                             const int accessID) {
  DAWN_ASSERT(ExprToAccessIDMap_.count(expr));
  ExprToAccessIDMap_[expr] = accessID;
}

void StencilInstantiation::removeStencilFunctionInstantiation(
    const std::shared_ptr<StencilFunCallExpr>& expr,
    std::shared_ptr<StencilFunctionInstantiation> callerStencilFunctionInstantiation) {

  std::shared_ptr<StencilFunctionInstantiation> func = nullptr;

  if(callerStencilFunctionInstantiation) {
    func = callerStencilFunctionInstantiation->getStencilFunctionInstantiation(expr);
    callerStencilFunctionInstantiation->removeStencilFunctionInstantiation(expr);
  } else {
    func = getStencilFunctionInstantiation(expr);
    ExprToStencilFunctionInstantiationMap_.erase(expr);
    nameToStencilFunctionInstantiationMap_.erase(expr->getCallee());
  }

  for(auto it = stencilFunctionInstantiations_.begin();
      it != stencilFunctionInstantiations_.end();) {
    if(*it == func)
      it = stencilFunctionInstantiations_.erase(it);
    else
      ++it;
  }
}

const std::shared_ptr<StencilFunctionInstantiation>
StencilInstantiation::getStencilFunctionInstantiation(
    const std::shared_ptr<StencilFunCallExpr>& expr) const {
  auto it = ExprToStencilFunctionInstantiationMap_.find(expr);
  DAWN_ASSERT_MSG(it != ExprToStencilFunctionInstantiationMap_.end(), "Invalid stencil function");
  return it->second;
}

const std::shared_ptr<StencilFunctionInstantiation>
StencilInstantiation::getStencilFunctionInstantiation(const std::string stencilFunName) const {
  auto it = nameToStencilFunctionInstantiationMap_.find(stencilFunName);
  DAWN_ASSERT_MSG(it != nameToStencilFunctionInstantiationMap_.end(), "Invalid stencil function");
  return it->second;
}

std::shared_ptr<StencilFunctionInstantiation>
StencilInstantiation::getStencilFunctionInstantiationCandidate(
    const std::shared_ptr<StencilFunCallExpr>& expr) {
  auto it = std::find_if(stencilFunInstantiationCandidate_.begin(),
                         stencilFunInstantiationCandidate_.end(),
                         [&](std::pair<std::shared_ptr<StencilFunctionInstantiation>,
                                       StencilFunctionInstantiationCandidate> const& pair) {
                           return (pair.first->getExpression() == expr);
                         });
  DAWN_ASSERT_MSG((it != stencilFunInstantiationCandidate_.end()),
                  "stencil function candidate not found");

  return it->first;
}

bool StencilInstantiation::hasStencilFunctionInstantiationCandidate(
    const std::string stencilFunName) const {
  auto it = std::find_if(stencilFunInstantiationCandidate_.begin(),
                         stencilFunInstantiationCandidate_.end(),
                         [&](std::pair<std::shared_ptr<StencilFunctionInstantiation>,
                                       StencilFunctionInstantiationCandidate> const& pair) {
                           return (pair.first->getExpression()->getCallee() == stencilFunName);
                         });
  return (it != stencilFunInstantiationCandidate_.end());
}

bool StencilInstantiation::hasStencilFunctionInstantiation(const std::string stencilFunName) const {
  auto it = nameToStencilFunctionInstantiationMap_.find(stencilFunName);
  return it != nameToStencilFunctionInstantiationMap_.end();
}

std::shared_ptr<StencilFunctionInstantiation>
StencilInstantiation::getStencilFunctionInstantiationCandidate(const std::string stencilFunName) {
  auto it = std::find_if(stencilFunInstantiationCandidate_.begin(),
                         stencilFunInstantiationCandidate_.end(),
                         [&](std::pair<std::shared_ptr<StencilFunctionInstantiation>,
                                       StencilFunctionInstantiationCandidate> const& pair) {
                           return (pair.first->getExpression()->getCallee() == stencilFunName);
                         });
  DAWN_ASSERT_MSG((it != stencilFunInstantiationCandidate_.end()),
                  "stencil function candidate not found");

  return it->first;
}

std::shared_ptr<StencilFunctionInstantiation> StencilInstantiation::cloneStencilFunctionCandidate(
    const std::shared_ptr<StencilFunctionInstantiation>& stencilFun, std::string functionName) {
  DAWN_ASSERT(stencilFunInstantiationCandidate_.count(stencilFun));
  auto stencilFunClone = std::make_shared<StencilFunctionInstantiation>(*stencilFun);

  auto stencilFunExpr =
      std::dynamic_pointer_cast<StencilFunCallExpr>(stencilFun->getExpression()->clone());
  stencilFunExpr->setCallee(functionName);

  auto sirStencilFun = std::make_shared<sir::StencilFunction>(*(stencilFun->getStencilFunction()));
  sirStencilFun->Name = functionName;

  stencilFunClone->setExpression(stencilFunExpr);
  stencilFunClone->setStencilFunction(sirStencilFun);

  stencilFunInstantiationCandidate_.emplace(stencilFunClone,
                                            stencilFunInstantiationCandidate_[stencilFun]);
  return stencilFunClone;
}

std::unordered_map<std::shared_ptr<StencilFunCallExpr>,
                   std::shared_ptr<StencilFunctionInstantiation>>&
StencilInstantiation::getExprToStencilFunctionInstantiationMap() {
  return ExprToStencilFunctionInstantiationMap_;
}

const std::unordered_map<std::shared_ptr<StencilFunCallExpr>,
                         std::shared_ptr<StencilFunctionInstantiation>>&
StencilInstantiation::getExprToStencilFunctionInstantiationMap() const {
  return ExprToStencilFunctionInstantiationMap_;
}

std::shared_ptr<StencilFunctionInstantiation>
StencilInstantiation::makeStencilFunctionInstantiation(
    const std::shared_ptr<StencilFunCallExpr>& expr,
    const std::shared_ptr<sir::StencilFunction>& SIRStencilFun, const std::shared_ptr<AST>& ast,
    const Interval& interval,
    const std::shared_ptr<StencilFunctionInstantiation>& curStencilFunctionInstantiation) {

  std::shared_ptr<StencilFunctionInstantiation> stencilFun =
      std::make_shared<StencilFunctionInstantiation>(this, expr, SIRStencilFun, ast, interval,
                                                     curStencilFunctionInstantiation != nullptr);

  stencilFunInstantiationCandidate_.emplace(
      stencilFun, StencilFunctionInstantiationCandidate{curStencilFunctionInstantiation});

  return stencilFun;
}

void StencilInstantiation::insertExprToStencilFunction(
    std::shared_ptr<StencilFunctionInstantiation> stencilFun) {
  ExprToStencilFunctionInstantiationMap_.emplace(stencilFun->getExpression(), stencilFun);
  nameToStencilFunctionInstantiationMap_.emplace(stencilFun->getExpression()->getCallee(),
                                                 stencilFun);
}

void StencilInstantiation::finalizeStencilFunctionSetup(
    std::shared_ptr<StencilFunctionInstantiation> stencilFun) {

  DAWN_ASSERT(stencilFunInstantiationCandidate_.count(stencilFun));
  stencilFun->closeFunctionBindings();

  // We take the candidate to stencil function and placed it in the stencil function instantiations
  // container
  StencilFunctionInstantiationCandidate candidate = stencilFunInstantiationCandidate_[stencilFun];

  // map of expr to stencil function instantiation is updated
  if(candidate.callerStencilFunction_) {
    candidate.callerStencilFunction_->insertExprToStencilFunction(stencilFun);
  } else {
    insertExprToStencilFunction(stencilFun);
  }

  stencilFun->update();

  stencilFunctionInstantiations_.push_back(stencilFun);
  // we remove the candidate to stencil function
  stencilFunInstantiationCandidate_.erase(stencilFun);
}

std::unordered_map<std::shared_ptr<StencilCallDeclStmt>, int>&
StencilInstantiation::getStencilCallToStencilIDMap() {
  return StencilCallToStencilIDMap_;
}

const std::unordered_map<std::shared_ptr<StencilCallDeclStmt>, int>&
StencilInstantiation::getStencilCallToStencilIDMap() const {
  return StencilCallToStencilIDMap_;
}

std::unordered_map<int, std::shared_ptr<StencilCallDeclStmt>>&
StencilInstantiation::getIDToStencilCallMap() {
  return IDToStencilCallMap_;
}

const std::unordered_map<int, std::shared_ptr<StencilCallDeclStmt>>&
StencilInstantiation::getIDToStencilCallMap() const {
  return IDToStencilCallMap_;
}

int StencilInstantiation::getStencilIDFromStmt(
    const std::shared_ptr<StencilCallDeclStmt>& stmt) const {
  auto it = StencilCallToStencilIDMap_.find(stmt);
  DAWN_ASSERT_MSG(it != StencilCallToStencilIDMap_.end(), "Invalid stencil call");
  return it->second;
}

std::unordered_map<std::string, int>& StencilInstantiation::getNameToAccessIDMap() {
  return NameToAccessIDMap_;
}

const std::unordered_map<std::string, int>& StencilInstantiation::getNameToAccessIDMap() const {
  return NameToAccessIDMap_;
}

std::unordered_map<int, std::string>& StencilInstantiation::getAccessIDToNameMap() {
  return AccessIDToNameMap_;
}

const std::unordered_map<int, std::string>& StencilInstantiation::getAccessIDToNameMap() const {
  return AccessIDToNameMap_;
}

std::unordered_map<int, std::string>& StencilInstantiation::getLiteralAccessIDToNameMap() {
  return LiteralAccessIDToNameMap_;
}
const std::unordered_map<int, std::string>&
StencilInstantiation::getLiteralAccessIDToNameMap() const {
  return LiteralAccessIDToNameMap_;
}

std::unordered_map<int, std::string>& StencilInstantiation::getStageIDToNameMap() {
  return StageIDToNameMap_;
}

const std::unordered_map<int, std::string>& StencilInstantiation::getStageIDToNameMap() const {
  return StageIDToNameMap_;
}

std::set<int>& StencilInstantiation::getFieldAccessIDSet() { return FieldAccessIDSet_; }

const std::set<int>& StencilInstantiation::getFieldAccessIDSet() const { return FieldAccessIDSet_; }

std::set<int>& StencilInstantiation::getGlobalVariableAccessIDSet() {
  return GlobalVariableAccessIDSet_;
}

const std::set<int>& StencilInstantiation::getGlobalVariableAccessIDSet() const {
  return GlobalVariableAccessIDSet_;
}

namespace {

/// @brief Get the orignal name of the field (or variable) given by AccessID and a list of
/// SourceLocations where this field (or variable) was accessed.
class OriginalNameGetter : public ASTVisitorForwarding {
  const StencilInstantiation* instantiation_;
  const int AccessID_;
  const bool captureLocation_;

  std::string name_;
  std::vector<SourceLocation> locations_;

public:
  OriginalNameGetter(const StencilInstantiation* instantiation, int AccessID, bool captureLocation)
      : instantiation_(instantiation), AccessID_(AccessID), captureLocation_(captureLocation) {}

  virtual void visit(const std::shared_ptr<VarDeclStmt>& stmt) override {
    if(instantiation_->getAccessIDFromStmt(stmt) == AccessID_) {
      name_ = stmt->getName();
      if(captureLocation_)
        locations_.push_back(stmt->getSourceLocation());
    }

    for(const auto& expr : stmt->getInitList())
      expr->accept(*this);
  }

  void visit(const std::shared_ptr<VarAccessExpr>& expr) override {
    if(instantiation_->getAccessIDFromExpr(expr) == AccessID_) {
      name_ = expr->getName();
      if(captureLocation_)
        locations_.push_back(expr->getSourceLocation());
    }
  }

  void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override {
    if(instantiation_->getAccessIDFromExpr(expr) == AccessID_) {
      name_ = expr->getValue();
      if(captureLocation_)
        locations_.push_back(expr->getSourceLocation());
    }
  }

  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {
    if(instantiation_->getAccessIDFromExpr(expr) == AccessID_) {
      name_ = expr->getName();
      if(captureLocation_)
        locations_.push_back(expr->getSourceLocation());
    }
  }

  std::pair<std::string, std::vector<SourceLocation>> getNameLocationPair() const {
    return std::make_pair(name_, locations_);
  }

  bool hasName() const { return !name_.empty(); }
  std::string getName() const { return name_; }
};

} // anonymous namespace

std::pair<std::string, std::vector<SourceLocation>>
StencilInstantiation::getOriginalNameAndLocationsFromAccessID(
    int AccessID, const std::shared_ptr<Stmt>& stmt) const {
  OriginalNameGetter orignalNameGetter(this, AccessID, true);
  stmt->accept(orignalNameGetter);
  return orignalNameGetter.getNameLocationPair();
}

std::string StencilInstantiation::getOriginalNameFromAccessID(int AccessID) const {
  OriginalNameGetter orignalNameGetter(this, AccessID, true);
  for(const auto& stencil : stencils_)
    for(const auto& multistage : stencil->getMultiStages())
      for(const auto& stage : multistage->getStages())
        for(const auto& doMethod : stage->getDoMethods())
          for(const auto& statementAccessesPair : doMethod->getStatementAccessesPairs()) {
            statementAccessesPair->getStatement()->ASTStmt->accept(orignalNameGetter);
            if(orignalNameGetter.hasName())
              return orignalNameGetter.getName();
          }

  // Best we can do...
  return getNameFromAccessID(AccessID);
}

namespace {

template <int Level>
struct PrintDescLine {
  PrintDescLine(const Twine& name) {
    std::cout << MakeIndent<Level>::value << format("\e[1;3%im", Level) << name.str() << "\n"
              << MakeIndent<Level>::value << "{\n\e[0m";
  }
  ~PrintDescLine() { std::cout << MakeIndent<Level>::value << format("\e[1;3%im}\n\e[0m", Level); }
};

} // anonymous namespace

void StencilInstantiation::dump() const {
  std::cout << "StencilInstantiation : " << getName() << "\n";

  for(std::size_t i = 0; i < stencils_.size(); ++i) {
    PrintDescLine<1> iline("Stencil_" + Twine(i));

    int j = 0;
    const auto& multiStages = stencils_[i]->getMultiStages();
    for(const auto& multiStage : multiStages) {
      PrintDescLine<2> jline(Twine("MultiStage_") + Twine(j) + " [" +
                             loopOrderToString(multiStage->getLoopOrder()) + "]");

      int k = 0;
      const auto& stages = multiStage->getStages();
      for(const auto& stage : stages) {
        PrintDescLine<3> kline(Twine("Stage_") + Twine(k));

        int l = 0;
        const auto& doMethods = stage->getDoMethods();
        for(const auto& doMethod : doMethods) {
          PrintDescLine<4> lline(Twine("Do_") + Twine(l) + " " +
                                 doMethod->getInterval().toString());

          const auto& statementAccessesPairs = doMethod->getStatementAccessesPairs();
          for(std::size_t m = 0; m < statementAccessesPairs.size(); ++m) {
            std::cout << "\e[1m"
                      << ASTStringifer::toString(statementAccessesPairs[m]->getStatement()->ASTStmt,
                                                 5 * DAWN_PRINT_INDENT)
                      << "\e[0m";
            std::cout << statementAccessesPairs[m]->getAccesses()->toString(this,
                                                                            6 * DAWN_PRINT_INDENT)
                      << "\n";
          }
          l += 1;
        }
        std::cout << "\e[1m" << std::string(4 * DAWN_PRINT_INDENT, ' ')
                  << "Extents: " << stage->getExtents() << std::endl
                  << "\e[0m";
        k += 1;
      }
      j += 1;
    }
  }
  std::cout.flush();
}

void StencilInstantiation::dumpAsJson(std::string filename, std::string passName) const {
  json::json jout;

  for(int i = 0; i < stencils_.size(); ++i) {
    json::json jStencil;

    int j = 0;
    const auto& multiStages = stencils_[i]->getMultiStages();
    for(const auto& multiStage : multiStages) {
      json::json jMultiStage;
      jMultiStage["LoopOrder"] = loopOrderToString(multiStage->getLoopOrder());

      int k = 0;
      const auto& stages = multiStage->getStages();
      for(const auto& stage : stages) {
        json::json jStage;

        int l = 0;
        for(const auto& doMethod : stage->getDoMethods()) {
          json::json jDoMethod;

          jDoMethod["Interval"] = doMethod->getInterval().toString();

          const auto& statementAccessesPairs = doMethod->getStatementAccessesPairs();
          for(std::size_t m = 0; m < statementAccessesPairs.size(); ++m) {
            jDoMethod["Stmt_" + std::to_string(m)] = ASTStringifer::toString(
                statementAccessesPairs[m]->getStatement()->ASTStmt, 0, false);
            jDoMethod["Accesses_" + std::to_string(m)] =
                statementAccessesPairs[m]->getAccesses()->reportAccesses(this);
          }

          jStage["Do_" + std::to_string(l++)] = jDoMethod;
        }

        jMultiStage["Stage_" + std::to_string(k++)] = jStage;
      }

      jStencil["MultiStage_" + std::to_string(j++)] = jMultiStage;
    }

    if(passName.empty())
      jout[getName()]["Stencil_" + std::to_string(i)] = jStencil;
    else
      jout[passName][getName()]["Stencil_" + std::to_string(i)] = jStencil;
  }

  std::ofstream fs(filename, std::ios::out | std::ios::trunc);
  if(!fs.is_open()) {
    DiagnosticsBuilder diag(DiagnosticsKind::Error, SourceLocation());
    diag << "file system error: cannot open file: " << filename;
    context_->getDiagnostics().report(diag);
  }

  fs << jout.dump(2) << std::endl;
  fs.close();
}

static std::string makeNameImpl(const char* prefix, const std::string& name, int AccessID) {
  return prefix + name + "_" + std::to_string(AccessID);
}

static std::string extractNameImpl(StringRef prefix, const std::string& name) {
  StringRef nameRef(name);

  // Remove leading `prefix`
  std::size_t leadingLocalPos = nameRef.find(prefix);
  nameRef = nameRef.drop_front(leadingLocalPos != StringRef::npos ? prefix.size() : 0);

  // Remove trailing `_X` where X is the AccessID
  std::size_t trailingAccessIDPos = nameRef.find_last_of('_');
  nameRef = nameRef.drop_back(
      trailingAccessIDPos != StringRef::npos ? nameRef.size() - trailingAccessIDPos : 0);

  return nameRef.empty() ? name : nameRef.str();
}

std::string StencilInstantiation::makeLocalVariablename(const std::string& name, int AccessID) {
  return makeNameImpl("__local_", name, AccessID);
}

std::string StencilInstantiation::makeTemporaryFieldname(const std::string& name, int AccessID) {
  return makeNameImpl("__tmp_", name, AccessID);
}

std::string StencilInstantiation::extractLocalVariablename(const std::string& name) {
  return extractNameImpl("__local_", name);
}

std::string StencilInstantiation::extractTemporaryFieldname(const std::string& name) {
  return extractNameImpl("__tmp_", name);
}

std::string StencilInstantiation::makeStencilCallCodeGenName(int StencilID) {
  return "__code_gen_" + std::to_string(StencilID);
}

bool StencilInstantiation::isStencilCallCodeGenName(const std::string& name) {
  return StringRef(name).startswith("__code_gen_");
}

const std::set<int>& StencilInstantiation::getCachedVariableSet() const {
  return CachedVariableSet_;
}

void StencilInstantiation::insertCachedVariable(int fieldID) {
  CachedVariableSet_.emplace(fieldID);
}

void StencilInstantiation::reportAccesses() const {
  // Stencil functions
  for(const auto& stencilFun : stencilFunctionInstantiations_) {
    const auto& statementAccessesPairs = stencilFun->getStatementAccessesPairs();

    for(std::size_t i = 0; i < statementAccessesPairs.size(); ++i) {
      std::cout << "\nACCESSES: line "
                << statementAccessesPairs[i]->getStatement()->ASTStmt->getSourceLocation().Line
                << ": "
                << statementAccessesPairs[i]->getCalleeAccesses()->reportAccesses(stencilFun.get())
                << "\n";
    }
  }

  // Stages
  for(const auto& stencil : stencils_)
    for(const auto& multistage : stencil->getMultiStages())
      for(const auto& stage : multistage->getStages()) {
        for(const auto& doMethod : stage->getDoMethods()) {
          const auto& statementAccessesPairs = doMethod->getStatementAccessesPairs();

          for(std::size_t i = 0; i < statementAccessesPairs.size(); ++i) {
            std::cout
                << "\nACCESSES: line "
                << statementAccessesPairs[i]->getStatement()->ASTStmt->getSourceLocation().Line
                << ": " << statementAccessesPairs[i]->getAccesses()->reportAccesses(this) << "\n";
          }
        }
      }
}

} // namespace dawn
