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
#include "dawn/IIR/ASTConverter.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/InstantiationHelper.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/AccessComputation.h"
#include "dawn/Optimizer/PassComputeStageExtents.h"
#include "dawn/Optimizer/PassSetStageName.h"
#include "dawn/Optimizer/PassTemporaryType.h"
#include "dawn/Optimizer/StatementMapper.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/STLExtras.h"
#include <stack>

namespace dawn {

namespace {

/// @brief Helper function to encapsulate conversion performed by ASTConverter from AST with sir
/// data to equal AST (cloned) but with iir data
std::shared_ptr<ast::AST> convertToIIRAST(const ast::AST& sirAST) {
  ASTConverter astConverter;
  sirAST.accept(astConverter);
  return std::make_shared<ast::AST>(
      std::dynamic_pointer_cast<iir::BlockStmt>(astConverter.getStmtMap().at(sirAST.getRoot())));
}

using namespace iir;
//===------------------------------------------------------------------------------------------===//
//     StencilDescStatementMapper
//===------------------------------------------------------------------------------------------===//

/// @brief Map the statements of the stencil description AST to a flat list of statements and
/// inline all calls to other stencils
class StencilDescStatementMapper : public iir::ASTVisitor {

  /// @brief Record of the current scope (each StencilCall will create a new scope)
  struct Scope : public NonCopyable {
    Scope(const std::string& name, ControlFlowDescriptor& controlFlowDescriptor,
          const std::vector<ast::StencilCall*>& stackTrace)
        : Name(name), ScopeDepth(0), controlFlowDescriptor_(controlFlowDescriptor),
          StackTrace(stackTrace) {}

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

    /// Current call stack of stencil calls (may be empty)
    std::vector<ast::StencilCall*> StackTrace;
  };

  const std::shared_ptr<iir::StencilInstantiation>& instantiation_;
  iir::StencilMetaInformation& metadata_;
  std::stack<std::shared_ptr<Scope>> scope_;

  sir::Stencil* sirStencil_;

  const std::vector<std::shared_ptr<sir::Stencil>>& stencils_;

  /// We replace the first VerticalRegionDeclStmt with a dummy node which signals code-gen that it
  /// should insert a call to the gridtools stencil here
  std::shared_ptr<iir::Stmt> stencilDescReplacement_;

  OptimizerContext& context_;

public:
  StencilDescStatementMapper(std::shared_ptr<iir::StencilInstantiation>& instantiation,
                             sir::Stencil* sirStencil,
                             const std::vector<std::shared_ptr<sir::Stencil>>& stencils,
                             const sir::GlobalVariableMap& globalVariableMap,
                             OptimizerContext& context)
      : instantiation_(instantiation), metadata_(instantiation->getMetaData()),
        sirStencil_(sirStencil), stencils_(stencils), context_(context) {
    DAWN_ASSERT(instantiation);
    // Create the initial scope
    scope_.push(std::make_shared<Scope>(sirStencil_->Name,
                                        instantiation_->getIIR()->getControlFlowDescriptor(),
                                        std::vector<ast::StencilCall*>()));
    scope_.top()->LocalFieldnameToAccessIDMap = metadata_.getNameToAccessIDMap();

    // We add all global variables which have constant values
    for(const auto& keyValuePair : globalVariableMap) {
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

  /// @brief Create a new stencil in the instantiation and prepare the replacement node for the
  /// next VerticalRegionDeclStmt
  /// @see tryReplaceVerticalRegionDeclStmt
  void makeNewStencil() {
    int StencilID = instantiation_->nextUID();
    instantiation_->getIIR()->insertChild(
        std::make_unique<Stencil>(metadata_, sirStencil_->Attributes, StencilID),
        instantiation_->getIIR());
    // We create a paceholder stencil-call for CodeGen to know wehere we need to insert calls to
    // this stencil
    auto placeholderStencil = std::make_shared<ast::StencilCall>(
        InstantiationHelper::makeStencilCallCodeGenName(StencilID));
    auto stencilCallDeclStmt = iir::makeStencilCallDeclStmt(placeholderStencil);

    // Register the call and set it as a replacement for the next vertical region
    metadata_.addStencilCallStmt(stencilCallDeclStmt, StencilID);
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
    if(scope_.top()->ScopeDepth == 1) {
      stencilDescReplacement_->getData<iir::IIRStmtData>().StackTrace =
          std::make_optional(scope_.top()->StackTrace);
      scope_.top()->controlFlowDescriptor_.insertStmt(stencilDescReplacement_);
    } else {

      // We need to replace the VerticalRegionDeclStmt in the current statement
      iir::replaceOldStmtWithNewStmtInStmt(
          scope_.top()->controlFlowDescriptor_.getStatements().back(), stencilDescNode,
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
      statement->accept(remover);
  }

  /// @brief Push back a new statement to the end of the current statement list
  void pushBackStatement(const std::shared_ptr<iir::Stmt>& stmt) {
    stmt->getData<iir::IIRStmtData>().StackTrace = std::make_optional(scope_.top()->StackTrace);
    scope_.top()->controlFlowDescriptor_.insertStmt(stmt);
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
              scope_.top()->controlFlowDescriptor_.getStatements().back(), stmt,
              stmt->getThenStmt());
          stmt->getThenStmt()->accept(*this);
        } else if(stmt->hasElse()) {
          // Replace the if-statement with the else-block
          iir::replaceOldStmtWithNewStmtInStmt(
              scope_.top()->controlFlowDescriptor_.getStatements().back(), stmt,
              stmt->getElseStmt());
          stmt->getElseStmt()->accept(*this);
        } else {
          // Replace the if-statement with a void `0`
          auto voidExpr = std::make_shared<iir::LiteralAccessExpr>("0", BuiltinTypeID::Float);
          auto voidStmt = iir::makeExprStmt(voidExpr);
          int AccessID = -instantiation_->nextUID();
          metadata_.insertAccessOfType(iir::FieldAccessType::FAT_Literal, AccessID, "0");
          metadata_.insertExprToAccessID(voidExpr, AccessID);
          iir::replaceOldStmtWithNewStmtInStmt(
              scope_.top()->controlFlowDescriptor_.getStatements().back(), stmt, voidStmt);
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

    int AccessID = metadata_.addStmt(context_.getOptions().KeepVarnames, stmt);

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
    std::unique_ptr<MultiStage> multiStage = std::make_unique<MultiStage>(
        metadata_, verticalRegion->LoopOrder == sir::VerticalRegion::LK_Forward
                       ? LoopOrderKind::LK_Forward
                       : LoopOrderKind::LK_Backward);
    std::unique_ptr<Stage> stage =
        std::make_unique<Stage>(metadata_, instantiation_->nextUID(), interval);

    DAWN_LOG(INFO) << "Processing vertical region at " << verticalRegion->Loc;

    // Here we convert the AST of the vertical region to a flat list of statements of the stage.
    // Further, we instantiate all referenced stencil functions.
    DAWN_LOG(INFO) << "Inserting statements ... ";
    DoMethod& doMethod = stage->getSingleDoMethod();
    // TODO move iterators of IIRNode to const getChildren, when we pass here begin, end instead

    StatementMapper statementMapper(instantiation_.get(), context_, scope_.top()->StackTrace,
                                    doMethod, doMethod.getInterval(),
                                    scope_.top()->LocalFieldnameToAccessIDMap, nullptr);
    ast->accept(statementMapper);
    DAWN_LOG(INFO) << "Inserted " << doMethod.getChildren().size() << " statements";

    if(context_.getDiagnostics().hasErrors())
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
    ast::StencilCall* stencilCall = stmt->getStencilCall().get();

    tryReplaceStencilDescStmt(stmt);

    DAWN_LOG(INFO) << "Processing stencil call to `" << stencilCall->Callee << "` at "
                   << stencilCall->Loc;

    // Prepare a new scope for the stencil call
    std::shared_ptr<Scope>& curScope = scope_.top();
    std::shared_ptr<Scope> candiateScope = std::make_shared<Scope>(
        curScope->Name, curScope->controlFlowDescriptor_, curScope->StackTrace);

    // Variables are inherited from the parent scope (note that this *needs* to be a copy as we
    // cannot modify the parent scope)
    candiateScope->VariableMap = curScope->VariableMap;

    // Record the call
    candiateScope->StackTrace.push_back(stencilCall);

    // Get the sir::Stencil from the callee name
    auto stencilIt = std::find_if(
        stencils_.begin(), stencils_.end(),
        [&](const std::shared_ptr<sir::Stencil>& s) { return s->Name == stencilCall->Callee; });
    DAWN_ASSERT(stencilIt != stencils_.end());
    sir::Stencil& stencil = **stencilIt;

    // We need less or an equal amount of args as temporaries are added implicitly
    DAWN_ASSERT(stencilCall->Args.size() <= stencil.Fields.size());

    // Map the field arguments
    for(std::size_t stencilArgIdx = 0, stencilCallArgIdx = 0; stencilArgIdx < stencil.Fields.size();
        ++stencilArgIdx) {

      int AccessID = 0;
      if(stencil.Fields[stencilArgIdx]->IsTemporary) {
        // We add a new temporary field for each temporary field argument
        AccessID = metadata_.addTmpField(iir::FieldAccessType::FAT_StencilTemporary,
                                         stencil.Fields[stencilArgIdx]->Name, {1, 1, 1});
      } else {
        AccessID = curScope->LocalFieldnameToAccessIDMap.at(stencilCall->Args[stencilCallArgIdx]);
        stencilCallArgIdx++;
      }

      candiateScope->LocalFieldnameToAccessIDMap.emplace(stencil.Fields[stencilArgIdx]->Name,
                                                         AccessID);
    }

    // Process the stencil description AST of the callee.
    scope_.push(candiateScope);

    // Convert the AST to have an AST with iir data and visit it
    convertToIIRAST(*stencil.StencilDescAst)->accept(*this);

    scope_.pop();

    DAWN_LOG(INFO) << "Done processing stencil call to `" << stencilCall->Callee << "`";
  }

  void visit(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt) override {
    if(instantiation_->insertBoundaryConditions(stmt->getFields()[0], stmt) == false)
      DAWN_ASSERT_MSG(false, "Boundary Condition specified twice for the same field");
    //      if(instantiation_->insertBoundaryConditions(stmt->getFields()[0]->Name, stmt) == false)
    //      DAWN_ASSERT_MSG(false, "Boundary Condition specified twice for the same field");
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
        DAWN_ASSERT_MSG(value.has_value(), "constant global variable with no value");

        auto newExpr = std::make_shared<dawn::LiteralAccessExpr>(
            value.toString(), sir::Value::typeToBuiltinTypeID(value.getType()));
        iir::replaceOldExprWithNewExprInStmt(
            scope_.top()->controlFlowDescriptor_.getStatements().back(), expr, newExpr);

        int AccessID = instantiation_->nextUID();
        metadata_.insertAccessOfType(iir::FieldAccessType::FAT_Literal, -AccessID,
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

  void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override {}
};
} // namespace

OptimizerContext::OptimizerContext(DiagnosticsEngine& diagnostics, OptimizerContextOptions options,
                                   const std::shared_ptr<SIR>& SIR)
    : diagnostics_(diagnostics), options_(options), SIR_(SIR) {
  DAWN_LOG(INFO) << "Intializing OptimizerContext ... ";
}
bool OptimizerContext::fillIIRFromSIR(
    std::shared_ptr<iir::StencilInstantiation> stencilInstantiation,
    const std::shared_ptr<sir::Stencil> SIRStencil, const std::shared_ptr<SIR> fullSIR) {
  DAWN_LOG(INFO) << "Intializing StencilInstantiation of `" << SIRStencil->Name << "`";
  DAWN_ASSERT_MSG(SIRStencil, "Stencil does not exist");
  auto& metadata = stencilInstantiation->getMetaData();
  metadata.setStencilname(SIRStencil->Name);
  metadata.setFileName(fullSIR->Filename);
  metadata.setStencilLocation(SIRStencil->Loc);

  // Map the fields of the "main stencil" to unique IDs (which are used in the access maps to
  // indentify the field).
  for(const auto& field : SIRStencil->Fields) {
    metadata.addField((field->IsTemporary ? iir::FieldAccessType::FAT_StencilTemporary
                                          : iir::FieldAccessType::FAT_APIField),
                      field->Name, field->fieldDimensions);
  }

  StencilDescStatementMapper stencilDeclMapper(stencilInstantiation, SIRStencil.get(),
                                               fullSIR->Stencils, *fullSIR->GlobalVariableMap,
                                               *this);

  //  Converting to AST with iir data
  auto AST = convertToIIRAST(*SIRStencil->StencilDescAst);
  AST->accept(stencilDeclMapper);

  //  Cleanup the `stencilDescStatements` and remove the empty stencils which may have been inserted
  stencilDeclMapper.cleanupStencilDeclAST();

  //  // Repair broken references to temporaries i.e promote them to real fields
  PassTemporaryType::fixTemporariesSpanningMultipleStencils(
      stencilInstantiation.get(), stencilInstantiation->getIIR()->getChildren());

  if(getOptions().ReportAccesses) {
    stencilInstantiation->reportAccesses();
  }

  for(const auto& MS : iterateIIROver<MultiStage>(*(stencilInstantiation->getIIR()))) {
    MS->update(NodeUpdateType::levelAndTreeAbove);
  }
  DAWN_LOG(INFO) << "Done initializing StencilInstantiation";

  // Iterate all statements (top -> bottom)
  for(const auto& stagePtr : iterateIIROver<iir::Stage>(*(stencilInstantiation->getIIR()))) {
    iir::Stage& stage = *stagePtr;
    for(const auto& doMethod : stage.getChildren()) {
      doMethod->update(iir::NodeUpdateType::level);
    }
    stage.update(iir::NodeUpdateType::level);
  }
  for(const auto& MSPtr : iterateIIROver<iir::Stage>(*(stencilInstantiation->getIIR()))) {
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

const OptimizerContext::OptimizerContextOptions& OptimizerContext::getOptions() const {
  return options_;
}

OptimizerContext::OptimizerContextOptions& OptimizerContext::getOptions() { return options_; }

void OptimizerContext::fillIIR() {
  DAWN_ASSERT(SIR_);
  std::vector<std::shared_ptr<sir::StencilFunction>> iirStencilFunctions;
  // Convert the asts of sir::StencilFunctions to iir
  std::transform(SIR_->StencilFunctions.begin(), SIR_->StencilFunctions.end(),
                 std::back_inserter(iirStencilFunctions),
                 [&](const std::shared_ptr<sir::StencilFunction>& sirSF) {
                   auto iirSF = std::make_shared<sir::StencilFunction>(*sirSF);
                   for(auto& ast : iirSF->Asts)
                     ast = convertToIIRAST(*ast);

                   return iirSF;
                 });

  for(const auto& stencil : SIR_->Stencils) {
    DAWN_ASSERT(stencil);
    if(!stencil->Attributes.has(sir::Attr::AK_NoCodeGen)) {
      stencilInstantiationMap_.insert(
          std::make_pair(stencil->Name, std::make_shared<iir::StencilInstantiation>(
                                            *getSIR()->GlobalVariableMap, iirStencilFunctions)));
      fillIIRFromSIR(stencilInstantiationMap_.at(stencil->Name), stencil, SIR_);
    } else {
      DAWN_LOG(INFO) << "Skipping processing of `" << stencil->Name << "`";
    }
  }
}

bool OptimizerContext::restoreIIR(std::string const& name,
                                  std::shared_ptr<iir::StencilInstantiation> stencilInstantiation) {
  auto& metadata = stencilInstantiation->getMetaData();
  metadata.setStencilname(stencilInstantiation->getName());
  metadata.setFileName("<unknown>");

  stencilInstantiationMap_.insert(std::make_pair(name, stencilInstantiation));

  for(const auto& MS : iterateIIROver<MultiStage>(*(stencilInstantiation->getIIR()))) {
    MS->update(NodeUpdateType::levelAndTreeAbove);
  }
  DAWN_LOG(INFO) << "Done initializing StencilInstantiation";

  // Iterate all statements (top -> bottom)
  for(const auto& stagePtr : iterateIIROver<iir::Stage>(*(stencilInstantiation->getIIR()))) {
    iir::Stage& stage = *stagePtr;
    for(const auto& doMethod : stage.getChildren()) {
      doMethod->update(iir::NodeUpdateType::level);
    }
    stage.update(iir::NodeUpdateType::level);
  }
  for(const auto& MSPtr : iterateIIROver<iir::Stage>(*(stencilInstantiation->getIIR()))) {
    MSPtr->update(iir::NodeUpdateType::levelAndTreeAbove);
  }

  // fix extents of stages since they are not stored in the iir but computed from the accesses
  // contained in the DoMethods
  checkAndPushBack<PassSetStageName>();
  checkAndPushBack<PassComputeStageExtents>();
  if(!getPassManager().runAllPassesOnStecilInstantiation(*this, stencilInstantiation))
    return false;

  return true;
}

} // namespace dawn
