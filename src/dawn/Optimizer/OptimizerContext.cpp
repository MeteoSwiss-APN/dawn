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
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/IIR/DependencyGraphStage.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/AccessComputation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassTemporaryType.h"
#include "dawn/Optimizer/Renaming.h"
#include "dawn/Optimizer/StatementMapper.h"
#include "dawn/SIR/ASTUtil.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/SIR/SIR.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/STLExtras.h"
#include "dawn/Support/StringUtil.h"
#include "dawn/Support/Unreachable.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stack>

namespace dawn {

namespace {
using namespace iir;
//===------------------------------------------------------------------------------------------===//
//     StencilDescStatementMapper
//===------------------------------------------------------------------------------------------===//

/// @brief Map the statements of the stencil description AST to a flat list of statements and
/// inline all calls to other stencils
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

  std::unique_ptr<iir::IIR> iir_;
  std::stack<std::shared_ptr<Scope>> scope_;

  sir::Stencil* sirStencil_;

  const std::shared_ptr<SIR> sir_;

  /// We replace the first VerticalRegionDeclStmt with a dummy node which signals code-gen that it
  /// should insert a call to the gridtools stencil here
  std::shared_ptr<Stmt> stencilDescReplacement_;

public:
  StencilDescStatementMapper(std::unique_ptr<iir::IIR>& iir, sir::Stencil* sirStencil,
                             const std::shared_ptr<SIR>& sir)
      : iir_(std::move(iir)), sirStencil_(sirStencil), sir_(sir) {
    DAWN_ASSERT(iir_);
    // Create the initial scope
    auto b = iir->getMetaData()->getStencilDescStatements();
    std::cout << "1.0" << std::endl;
    scope_.push(
        std::make_shared<Scope>(sirStencil_->Name, iir->getMetaData()->getStencilDescStatements()));
    scope_.top()->LocalFieldnameToAccessIDMap = iir->getMetaData()->getNameToAccessIDMap();

    // We add all global variables which have constant values
    for(const auto& keyValuePair : iir->getMetaData()->getGlobalVariableMap()) {
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
    std::cout << "1.1" << std::endl;
    // We start with a single stencil
    makeNewStencil();
  }

  /// @brief Create a new stencil in the instantiation and prepare the replacement node for the
  /// next VerticalRegionDeclStmt
  /// @see tryReplaceVerticalRegionDeclStmt
  void makeNewStencil() {
    int StencilID = iir_->getMetaData()->nextUID();
    iir_->insertChild(make_unique<Stencil>(iir_.get(), sirStencil_->Attributes, StencilID), iir_);
    // We create a paceholder stencil-call for CodeGen to know wehere we need to insert calls to
    // this stencil
    auto placeholderStencil = std::make_shared<sir::StencilCall>(
        iir_->getMetaData()->makeStencilCallCodeGenName(StencilID));
    auto stencilCallDeclStmt = std::make_shared<StencilCallDeclStmt>(placeholderStencil);

    // Register the call and set it as a replacement for the next vertical region
    iir_->getMetaData()->getStencilCallToStencilIDMap().emplace(stencilCallDeclStmt, StencilID);
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
    for(auto it = iir_->childrenBegin(); it != iir_->childrenEnd();) {
      Stencil& stencil = **it;
      if(stencil.isEmpty())
        it = iir_->childrenErase(it);
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
        // with either the then-block or the else-block or in case we evaluted to `false` and
        // there
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
          int AccessID = -iir_->getMetaData()->nextUID();
          iir_->getMetaData()->getLiteralAccessIDToNameMap().emplace(AccessID, "0");
          iir_->getMetaData()->mapExprToAccessID(voidExpr, AccessID);
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
    int AccessID = iir_->getMetaData()->nextUID();

    std::string globalName;
    DAWN_ASSERT_MSG(false, "impelent context access");
    //    if(context_->getOptions().KeepVarnames)
    if(false)
      globalName = stmt->getName();
    else
      globalName = StencilInstantiation::makeLocalVariablename(stmt->getName(), AccessID);

    iir_->getMetaData()->setAccessIDNamePair(AccessID, globalName);
    iir_->getMetaData()->getStmtToAccessIDMap().emplace(stmt, AccessID);

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
    std::unique_ptr<MultiStage> multiStage = make_unique<MultiStage>(
        iir_.get(), verticalRegion->LoopOrder == sir::VerticalRegion::LK_Forward
                        ? LoopOrderKind::LK_Forward
                        : LoopOrderKind::LK_Backward);
    std::unique_ptr<Stage> stage =
        make_unique<Stage>(nullptr, iir_->getMetaData()->nextUID(), interval);

    DAWN_LOG(INFO) << "Processing vertical region at " << verticalRegion->Loc;

    // Here we convert the AST of the vertical region to a flat list of statements of the stage.
    // Further, we instantiate all referenced stencil functions.
    DAWN_LOG(INFO) << "Inserting statements ... ";
    DoMethod& doMethod = stage->getSingleDoMethod();
    // TODO move iterators of IIRNode to const getChildren, when we pass here begin, end instead

    StatementMapper statementMapper(sir_, iir_.get(), scope_.top()->StackTrace, doMethod,
                                    doMethod.getInterval(),
                                    scope_.top()->LocalFieldnameToAccessIDMap, nullptr);
    ast->accept(statementMapper);
    DAWN_LOG(INFO) << "Inserted " << doMethod.getChildren().size() << " statements";

    if(iir_->getDiagnostics().hasErrors())
      return;

    // Here we compute the *actual* access of each statement and associate access to the AccessIDs
    // we set previously.
    DAWN_LOG(INFO) << "Filling accesses ...";
    computeAccesses(iir_.get(), doMethod.getChildren());

    // Now, we compute the fields of each stage (this will give us the IO-Policy of the fields)
    stage->update(iir::NodeUpdateType::level);

    // Put the stage into a separate MultiStage ...
    multiStage->insertChild(std::move(stage));

    // ... and append the MultiStages of the current stencil
    const auto& stencil = iir_->getChildren().back();
    stencil->insertChild(std::move(multiStage));
  }

  void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override {
    DAWN_ASSERT_MSG(false, "not yet implemented");
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
        AccessID = iir_->getMetaData()->nextUID();
        iir_->getMetaData()->setAccessIDNamePairOfField(
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

    // As we *may* modify the AST we better make a copy here otherwise we get funny surprises if
    // we call this stencil multiple times ...
    stencil.StencilDescAst->clone()->accept(*this);

    scope_.pop();

    DAWN_LOG(INFO) << "Done processing stencil call to `" << stencilCall->Callee << "`";
  }

  void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) override {
    DAWN_ASSERT_MSG(false, "BOUNDARY CONDTITION NOT IMPLEMENTED");
    //      if(instantiation_->insertBoundaryConditions(stmt->getFields()[0]->Name, stmt) == false)
    //      DAWN_ASSERT_MSG(false, "Boundary Condition specified twice for the same field");
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

      const auto& value = iir_->getMetaData()->getGlobalVariableValue(varname);
      if(value.isConstexpr()) {
        // Replace the variable access with the actual value
        DAWN_ASSERT_MSG(!value.empty(), "constant global variable with no value");

        auto newExpr = std::make_shared<dawn::LiteralAccessExpr>(
            value.toString(), sir::Value::typeToBuiltinTypeID(value.getType()));
        replaceOldExprWithNewExprInStmt(scope_.top()->Statements.back()->ASTStmt, expr, newExpr);

        int AccessID = iir_->getMetaData()->nextUID();
        iir_->getMetaData()->getLiteralAccessIDToNameMap().emplace(AccessID, newExpr->getValue());
        iir_->getMetaData()->mapExprToAccessID(newExpr, AccessID);

      } else {
        int AccessID = 0;
        if(!iir_->getMetaData()->isGlobalVariable(varname)) {
          AccessID = iir_->getMetaData()->nextUID();
          iir_->getMetaData()->setAccessIDNamePairOfGlobalVariable(AccessID, varname);
        } else {
          AccessID = iir_->getMetaData()->getAccessIDFromName(varname);
        }

        iir_->getMetaData()->mapExprToAccessID(expr, AccessID);
      }

    } else {
      // Register the mapping between VarAccessExpr and AccessID.
      iir_->getMetaData()->mapExprToAccessID(expr,
                                             scope_.top()->LocalVarNameToAccessIDMap[varname]);

      // Resolve the index if this is an array access
      if(expr->isArrayAccess())
        expr->getIndex()->accept(*this);
    }
  }

  void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override {
    // Register a literal access (Note: the negative AccessID we assign!)
    int AccessID = -iir_->getMetaData()->nextUID();
    iir_->getMetaData()->getLiteralAccessIDToNameMap().emplace(AccessID, expr->getValue());
    iir_->getMetaData()->mapExprToAccessID(expr, AccessID);
  }

  void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {}
};
} // namespace anonymous

OptimizerContext::OptimizerContext(DiagnosticsEngine& diagnostics, Options& options,
                                   const std::shared_ptr<SIR>& SIR)
    : diagnostics_(diagnostics), options_(options), SIR_(SIR) {
  DAWN_LOG(INFO) << "Intializing OptimizerContext ... ";

  for(const auto& stencil : SIR_->Stencils)
    if(!stencil->Attributes.has(sir::Attr::AK_NoCodeGen)) {
      // this will become obsolete
      //{
      //      stencilInstantiationMap_.insert(std::make_pair(
      //          stencil->Name, std::make_shared<iir::StencilInstantiation>(this, stencil, SIR)));
      //}

      auto newIIR = make_unique<iir::IIR>();
      fillIIRFromSIR(newIIR, stencil, SIR_);
      stencilNameToIIRMap_.insert(std::make_pair(stencil->Name, std::move(newIIR)));

    } else {
      DAWN_LOG(INFO) << "Skipping processing of `" << stencil->Name << "`";
    }
}

bool OptimizerContext::fillIIRFromSIR(std::unique_ptr<iir::IIR>& iir,
                                      const std::shared_ptr<sir::Stencil> SIRStencil,
                                      const std::shared_ptr<SIR> fullSIR) {
  DAWN_LOG(INFO) << "Intializing StencilInstantiation of `" << SIRStencil->Name << "`";
  DAWN_ASSERT_MSG(SIRStencil, "Stencil does not exist");
  //    StencilDescStatementMapper stencilDeclMapper(this, getName(), stencilDescStatements_,
  //                                                 NameToAccessIDMap_);

  // Process the stencil description of the "main stencil"

  // Map the fields of the "main stencil" to unique IDs (which are used in the access maps to
  // indentify the field).
  for(const auto& field : SIRStencil->Fields) {
    int AccessID = iir->getMetaData()->nextUID();
    if(!field->IsTemporary) {
      iir->getMetaData()->getAPIFieldIDs().push_back(AccessID);
    }
    iir->getMetaData()->setAccessIDNamePairOfField(AccessID, field->Name, field->IsTemporary);
    iir->getMetaData()->getFieldIDToInitializedDimensionsMap().emplace(AccessID,
                                                                       field->fieldDimensions);
  }

  std::cout << "checkpoint 1" << std::endl;

    auto b = iir->getMetaData()->getStencilDescStatements();
  std::cout << "checkpoint 2" << std::endl;
  StencilDescStatementMapper stencilDeclMapper(iir, SIRStencil.get(), fullSIR);

  //  // We need to operate on a copy of the AST as we may modify the nodes inplace
  auto AST = SIRStencil->StencilDescAst->clone();
  AST->accept(stencilDeclMapper);

  std::cout << "checkpoint 3" << std::endl;

  //  // Cleanup the `stencilDescStatements` and remove the empty stencils which may have been
  //  inserted
  stencilDeclMapper.cleanupStencilDeclAST();

  std::cout << "checkpoint 4" << std::endl;

  //  // Repair broken references to temporaries i.e promote them to real fields
  PassTemporaryType::fixTemporariesSpanningMultipleStencils(iir.get(), iir->getChildren());

  std::cout << "checkpoint 5" << std::endl;

  if(iir->getOptions().ReportAccesses) {
      // WITTODO: reenable this
    //      iir->getMetaData()->reportAccesses();
  }

  for(const auto& MS : iterateIIROver<MultiStage>(*iir)) {
    MS->update(NodeUpdateType::levelAndTreeAbove);
  }
  std::cout << "checkpoint 6" << std::endl;

  std::cout << "printing iir of " << SIRStencil->Name << std::endl;
  iir->printTree();
  DAWN_LOG(INFO) << "Done initializing StencilInstantiation";

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
