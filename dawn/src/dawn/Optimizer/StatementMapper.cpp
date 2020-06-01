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
#include "dawn/Optimizer/StatementMapper.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/InstantiationHelper.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Support/Exception.h"
#include "dawn/Support/Logger.h"

namespace dawn {
StatementMapper::StatementMapper(
    iir::StencilInstantiation* instantiation, OptimizerContext& context,
    const std::vector<ast::StencilCall*>& stackTrace, iir::DoMethod& doMethod,
    const iir::Interval& interval,
    const std::unordered_map<std::string, int>& localFieldnameToAccessIDMap,
    const std::shared_ptr<iir::StencilFunctionInstantiation> stencilFunctionInstantiation)
    : instantiation_(instantiation), metadata_(instantiation->getMetaData()), context_(context),
      stackTrace_(stackTrace) {

  // Create the initial scope
  scope_.push(std::make_shared<Scope>(doMethod, interval, stencilFunctionInstantiation));
  scope_.top()->LocalFieldnameToAccessIDMap = localFieldnameToAccessIDMap;
}

StatementMapper::Scope* StatementMapper::getCurrentCandidateScope() {
  return (!scope_.top()->CandiateScopes.empty() ? scope_.top()->CandiateScopes.top().get()
                                                : nullptr);
}

void StatementMapper::appendNewStatement(const std::shared_ptr<iir::Stmt>& stmt) {
  stmt->getData<iir::IIRStmtData>().StackTrace = stackTrace_;
  if(scope_.top()->ScopeDepth == 1) {
    scope_.top()->doMethod_.getAST().push_back(std::shared_ptr<iir::Stmt>{stmt});
  }
}

void StatementMapper::visit(const std::shared_ptr<iir::BlockStmt>& stmt) {
  initializedWithBlockStmt_ = true;
  scope_.top()->ScopeDepth++;

  for(const auto& s : stmt->getStatements()) {
    s->accept(*this);
  }

  scope_.top()->ScopeDepth--;
}

void StatementMapper::visit(const std::shared_ptr<iir::LoopStmt>& stmt) {
  DAWN_ASSERT(initializedWithBlockStmt_);

  appendNewStatement(stmt);
  stmt->getBlockStmt()->accept(*this);
}

void StatementMapper::visit(const std::shared_ptr<iir::ExprStmt>& stmt) {
  DAWN_ASSERT(initializedWithBlockStmt_);
  appendNewStatement(stmt);
  stmt->getExpr()->accept(*this);
}

void StatementMapper::visit(const std::shared_ptr<iir::ReturnStmt>& stmt) {
  DAWN_ASSERT(initializedWithBlockStmt_);
  DAWN_ASSERT(scope_.top()->FunctionInstantiation);
  std::shared_ptr<const iir::StencilFunctionInstantiation> curFunc =
      scope_.top()->FunctionInstantiation;

  // We can only have 1 return statement
  if(curFunc->hasReturn()) {
    DAWN_DIAG(ERROR, metadata_.getFileName(), curFunc->getStencilFunction()->Loc)
        << "Multiple return-statement in stencil function '" << curFunc->getName() << "'";
  }
  scope_.top()->FunctionInstantiation->setReturn(true);

  appendNewStatement(stmt);
  stmt->getExpr()->accept(*this);
}

void StatementMapper::visit(const std::shared_ptr<iir::IfStmt>& stmt) {
  DAWN_ASSERT(initializedWithBlockStmt_);

  appendNewStatement(stmt);
  stmt->getCondExpr()->accept(*this);

  stmt->getThenStmt()->accept(*this);
  if(stmt->hasElse())
    stmt->getElseStmt()->accept(*this);
}

void StatementMapper::visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) {
  DAWN_ASSERT(initializedWithBlockStmt_);

  if(!stmt->getData<iir::VarDeclStmtData>().AccessID) {
    // TODO: this code is almost a duplicate of `StencilMetaInformation::addStmt()`, but here it
    // differs because it considers also the stencil function case. A common solution should be
    // found.

    // This is the first time we encounter this variable. We have to make sure the name is
    // not already used in another scope!
    int accessID = instantiation_->nextUID();

    std::string globalName;
    if(context_.getOptions().KeepVarnames)
      globalName = stmt->getName();
    else
      globalName = iir::InstantiationHelper::makeLocalVariablename(stmt->getName(), accessID);

    // We generate a new AccessID and insert it into the AccessMaps (using the global name)
    auto& function = scope_.top()->FunctionInstantiation;
    if(function) {
      function->getAccessIDToNameMap().emplace(accessID, globalName);
    } else {
      metadata_.addAccessIDNamePair(accessID, globalName);
      metadata_.addAccessIDToLocalVariableDataPair(accessID, iir::LocalVariableData{});
    }

    stmt->getData<iir::VarDeclStmtData>().AccessID = std::make_optional(accessID);

    // Add the mapping to the local scope
    scope_.top()->LocalVarNameToAccessIDMap.emplace(stmt->getName(), accessID);

    // Push back the statement and move on
    appendNewStatement(stmt);

    // Resolve the RHS
    for(const auto& expr : stmt->getInitList())
      expr->accept(*this);
  }
}

void StatementMapper::visit(const std::shared_ptr<iir::VerticalRegionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "VerticalRegionDeclStmt not allowed in this context");
}

void StatementMapper::visit(const std::shared_ptr<iir::StencilCallDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "StencilCallDeclStmt not allowed in this context");
}

void StatementMapper::visit(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "StencilCallDeclStmt not allowed in this context");
}

void StatementMapper::visit(const std::shared_ptr<iir::AssignmentExpr>& expr) {
  DAWN_ASSERT(initializedWithBlockStmt_);

  for(auto& s : expr->getChildren())
    s->accept(*this);
}

void StatementMapper::visit(const std::shared_ptr<iir::UnaryOperator>& expr) {
  DAWN_ASSERT(initializedWithBlockStmt_);

  for(auto& s : expr->getChildren())
    s->accept(*this);
}

void StatementMapper::visit(const std::shared_ptr<iir::ReductionOverNeighborExpr>& expr) {
  DAWN_ASSERT(initializedWithBlockStmt_);

  for(auto& s : expr->getChildren())
    s->accept(*this);
}

void StatementMapper::visit(const std::shared_ptr<iir::BinaryOperator>& expr) {
  DAWN_ASSERT(initializedWithBlockStmt_);

  for(auto& s : expr->getChildren())
    s->accept(*this);
}

void StatementMapper::visit(const std::shared_ptr<iir::TernaryOperator>& expr) {
  DAWN_ASSERT(initializedWithBlockStmt_);

  for(auto& s : expr->getChildren())
    s->accept(*this);
}

void StatementMapper::visit(const std::shared_ptr<iir::FunCallExpr>& expr) {
  DAWN_ASSERT(initializedWithBlockStmt_);

  for(auto& s : expr->getChildren())
    s->accept(*this);
}

void StatementMapper::visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) {
  DAWN_ASSERT(initializedWithBlockStmt_);

  // Find the referenced stencil function
  std::shared_ptr<iir::StencilFunctionInstantiation> stencilFun = nullptr;
  const iir::Interval& interval = scope_.top()->VerticalInterval;

  for(auto& iirStencilFun : instantiation_->getIIR()->getStencilFunctions()) {
    if(iirStencilFun->Name == expr->getCallee()) {

      std::shared_ptr<iir::AST> ast = nullptr;
      if(iirStencilFun->isSpecialized()) {
        // Select the correct overload
        ast = iirStencilFun->getASTOfInterval(interval.asSIRInterval());
        if(ast == nullptr) {
          throw SyntacticError(
              std::string("No viable Do-Method overload for stencil function call '") +
                  expr->getCallee() + "'",
              metadata_.getFileName(), expr->getSourceLocation());
        }
      } else {
        ast = iirStencilFun->Asts.front();
      }

      // Clone the AST s.t each stencil function has their own AST which is modifiable
      ast = ast->clone();

      // TODO decouple the funciton of stencil function instantiation from the statement mapper
      stencilFun = instantiation_->makeStencilFunctionInstantiation(
          expr, iirStencilFun, ast, interval, scope_.top()->FunctionInstantiation);
      break;
    }
  }
  DAWN_ASSERT(stencilFun);

  // If this is a nested function call (e.g the `bar` in `foo(bar(i+1, u))`) register the new
  // stencil function in the current stencil function
  if(Scope* candiateScope = getCurrentCandidateScope()) {
    candiateScope->FunctionInstantiation->setFunctionInstantiationOfArgField(
        candiateScope->ArgumentIndex, stencilFun);
    candiateScope->ArgumentIndex += 1;
  }

  // Create the scope of the stencil function
  scope_.top()->CandiateScopes.push(
      std::make_shared<Scope>(*(stencilFun->getDoMethod()), stencilFun->getInterval(), stencilFun));

  // Resolve the arguments
  for(auto& arg : expr->getArguments())
    arg->accept(*this);

  metadata_.finalizeStencilFunctionSetup(stencilFun);

  Scope* candiateScope = getCurrentCandidateScope();

  const auto& arguments = candiateScope->FunctionInstantiation->getArguments();

  for(std::size_t argIdx = 0; argIdx < arguments.size(); ++argIdx) {
    if(sir::Field* field = dyn_cast<sir::Field>(arguments[argIdx].get())) {
      int AccessID = candiateScope->FunctionInstantiation->getCallerAccessIDOfArgField(argIdx);
      candiateScope->LocalFieldnameToAccessIDMap.emplace(field->Name, AccessID);
    }
  }

  // Resolve the function
  scope_.push(scope_.top()->CandiateScopes.top());

  scope_.top()->FunctionInstantiation->getAST()->accept(*this);

  stencilFun->checkFunctionBindings();

  for(auto id : stencilFun->getAccessIDSetGlobalVariables()) {
    scope_.top()->LocalVarNameToAccessIDMap.emplace(stencilFun->getFieldNameFromAccessID(id), id);
  }

  scope_.pop();

  // We resolved the candiate function, move on ...
  scope_.top()->CandiateScopes.pop();
}

void StatementMapper::visit(const std::shared_ptr<iir::StencilFunArgExpr>& expr) {
  DAWN_ASSERT(initializedWithBlockStmt_);

  DAWN_ASSERT(!scope_.top()->CandiateScopes.empty());

  auto& function = scope_.top()->FunctionInstantiation;
  auto stencilFun = getCurrentCandidateScope()->FunctionInstantiation;
  auto& argumentIndex = getCurrentCandidateScope()->ArgumentIndex;
  bool needsLazyEval = expr->getArgumentIndex() != -1;

  if(stencilFun->isArgOffset(argumentIndex)) {
    // Argument is an offset
    stencilFun->setCallerOffsetOfArgOffset(
        argumentIndex, needsLazyEval
                           ? function->getCallerOffsetOfArgOffset(expr->getArgumentIndex())
                           : Array2i{{expr->getDimension(), expr->getOffset()}});
  } else {
    // Argument is a direction
    stencilFun->setCallerDimensionOfArgDirection(
        argumentIndex, needsLazyEval
                           ? function->getCallerDimensionOfArgDirection(expr->getArgumentIndex())
                           : expr->getDimension());
  }

  argumentIndex += 1;
}

void StatementMapper::visit(const std::shared_ptr<iir::VarAccessExpr>& expr) {
  DAWN_ASSERT(initializedWithBlockStmt_);

  auto& function = scope_.top()->FunctionInstantiation;
  const auto& varname = expr->getName();

  if(expr->isExternal()) {
    DAWN_ASSERT_MSG(!expr->isArrayAccess(), "global array access is not supported");

    const auto& value = instantiation_->getGlobalVariableValue(varname);
    if(value.isConstexpr()) {
      // Replace the variable access with the actual value
      DAWN_ASSERT_MSG(value.has_value(), "constant global variable with no value");

      auto newExpr = std::make_shared<iir::LiteralAccessExpr>(
          value.toString(), sir::Value::typeToBuiltinTypeID(value.getType()));
      iir::replaceOldExprWithNewExprInStmt(
          (*(scope_.top()->doMethod_.getAST().getStatements().rbegin())), expr, newExpr);

      // if a global is replaced by its value it becomes a de-facto literal negate access id
      int AccessID = -instantiation_->nextUID();

      metadata_.insertAccessOfType(iir::FieldAccessType::Literal, AccessID, newExpr->getValue());
      newExpr->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(AccessID);

    } else {
      int AccessID = 0;
      if(!metadata_.isAccessType(iir::FieldAccessType::GlobalVariable, varname)) {
        AccessID = metadata_.insertAccessOfType(iir::FieldAccessType::GlobalVariable, varname);
      } else {
        AccessID = metadata_.getAccessIDFromName(varname);
      }

      if(function)
        function->setAccessIDOfGlobalVariable(AccessID);

      expr->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(AccessID);
    }

  } else {
    // Register the mapping between VarAccessExpr and AccessID.
    expr->getData<iir::IIRAccessExprData>().AccessID =
        std::make_optional(scope_.top()->LocalVarNameToAccessIDMap[varname]);

    // Resolve the index if this is an array access
    if(expr->isArrayAccess())
      expr->getIndex()->accept(*this);
  }
}

void StatementMapper::visit(const std::shared_ptr<iir::LiteralAccessExpr>& expr) {
  DAWN_ASSERT(initializedWithBlockStmt_);

  // Register a literal access (Note: the negative AccessID we assign!)
  int AccessID = -instantiation_->nextUID();

  auto& function = scope_.top()->FunctionInstantiation;
  if(function)
    function->getLiteralAccessIDToNameMap().emplace(AccessID, expr->getValue());
  else
    metadata_.insertAccessOfType(iir::FieldAccessType::Literal, AccessID, expr->getValue());

  expr->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(AccessID);
}

void StatementMapper::visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) {
  DAWN_ASSERT(initializedWithBlockStmt_);
  // Register the mapping between FieldAccessExpr and AccessID
  int AccessID = scope_.top()->LocalFieldnameToAccessIDMap.at(expr->getName());

  auto& function = scope_.top()->FunctionInstantiation;

  expr->getData<iir::IIRAccessExprData>().AccessID = std::make_optional(AccessID);

  if(Scope* candiateScope = getCurrentCandidateScope()) {
    // We are currently traversing an argument list of a stencil function (create the mapping of
    // the arguments and compute the initial offset of the field)
    candiateScope->FunctionInstantiation->setCallerAccessIDOfArgField(candiateScope->ArgumentIndex,
                                                                      AccessID);
    candiateScope->FunctionInstantiation->setCallerInitialOffsetFromAccessID(
        AccessID, function ? function->evalOffsetOfFieldAccessExpr(expr) : expr->getOffset());
    candiateScope->ArgumentIndex += 1;
  }
}
} // namespace dawn
