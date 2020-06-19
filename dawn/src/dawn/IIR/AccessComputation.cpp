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

#include "dawn/IIR/AccessComputation.h"
#include "dawn/IIR/AST.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/Accesses.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/IIR/StencilMetaInformation.h"

#include <stack>

namespace dawn {

namespace {

/// @brief Compute and fill the access map of the given statement
class AccessMapper : public iir::ASTVisitor {
  const iir::StencilMetaInformation& metadata_;

  /// Keep track of the current statement
  struct CurrentStatement {
    CurrentStatement(const std::shared_ptr<iir::Stmt>& stmt) : Stmt(stmt), IfCondExpr(nullptr) {}

    /// Reference to the stmt we are currently working on.
    const std::shared_ptr<iir::Stmt>& Stmt;

    /// Access of the condition of the if-statement (this will be added to all subsequent accesses)
    /// For example:
    ///   if(true)
    ///     in = 5.0;   // Accesses:  W:in, R:5.0, R:true   (true is part of the read-access)
    std::shared_ptr<iir::Expr> IfCondExpr;
  };
  std::vector<std::unique_ptr<CurrentStatement>> curStatementStack_;

  /// List of caller and callee accesses. The first element is the primary accesses corresponding to
  /// the statement which was passed to the constructor. All other
  /// elements are the accesses of the children of the top-level statement.
  std::vector<iir::Accesses*> callerAccessesList_;
  std::vector<iir::Accesses*> calleeAccessesList_;

  /// Reference to the stencil function we are currently inside (if any)
  std::shared_ptr<iir::StencilFunctionInstantiation> stencilFun_;

  /// Reference to the current call of a stencil function if we are traversing an argument list
  struct StencilFunctionCallScope {
    StencilFunctionCallScope(
        std::shared_ptr<iir::StencilFunctionInstantiation> functionInstantiation)
        : FunctionInstantiation(functionInstantiation), ArgumentIndex(0) {}

    std::shared_ptr<iir::StencilFunctionInstantiation> FunctionInstantiation;
    int ArgumentIndex;
  };
  std::stack<std::unique_ptr<StencilFunctionCallScope>> stencilFunCalls_;

public:
  AccessMapper(const iir::StencilMetaInformation& metadata, const std::shared_ptr<iir::Stmt>& stmt,
               std::shared_ptr<iir::StencilFunctionInstantiation> stencilFun = nullptr)
      : metadata_(metadata), stencilFun_(stencilFun) {
    curStatementStack_.push_back(std::make_unique<CurrentStatement>(stmt));
  }

  /// @brief Get the stencil function instantiation from the `StencilFunCallExpr`
  std::shared_ptr<iir::StencilFunctionInstantiation>
  getStencilFunctionInstantiation(const std::shared_ptr<iir::StencilFunCallExpr>& expr) {
    return stencilFun_ ? stencilFun_->getStencilFunctionInstantiation(expr)
                       : metadata_.getStencilFunctionInstantiation(expr);
  }

  /// @brief Add a new access to the caller and callee and register it in the caller and callee
  /// accesses list. This will also add accesses to the children of the top-level statement
  void appendNewAccesses() {
    curStatementStack_.back()->Stmt->getData<iir::IIRStmtData>().CallerAccesses =
        std::make_optional(iir::Accesses());
    callerAccessesList_.emplace_back(
        &*curStatementStack_.back()->Stmt->getData<iir::IIRStmtData>().CallerAccesses);

    if(stencilFun_) {
      curStatementStack_.back()->Stmt->getData<iir::IIRStmtData>().CalleeAccesses =
          std::make_optional(iir::Accesses());
      calleeAccessesList_.emplace_back(
          &*curStatementStack_.back()->Stmt->getData<iir::IIRStmtData>().CalleeAccesses);
    } else
      curStatementStack_.back()->Stmt->getData<iir::IIRStmtData>().CalleeAccesses = std::nullopt;

    // Add all accesses of all parent if-cond expressions
    for(const auto& stmt : curStatementStack_)
      if(stmt->IfCondExpr)
        stmt->IfCondExpr->accept(*this);
  }

  /// @brief Pop the last added child access from the caller and callee accesses list
  void removeLastChildAccesses() {

    // The top-level stmt is never removed
    if(curStatementStack_.size() <= 1)
      return;

    callerAccessesList_.pop_back();
    if(stencilFun_)
      calleeAccessesList_.pop_back();
  }

  /// @brief Add a write extent/offset to the caller and callee accesses
  /// @{
  void mergeWriteOffset(const std::shared_ptr<iir::FieldAccessExpr>& field) {
    auto getOffset = [&](bool computeInitialOffset) {
      return stencilFun_ ? stencilFun_->evalOffsetOfFieldAccessExpr(field, computeInitialOffset)
                         : field->getOffset();
    };

    for(auto& callerAccesses : callerAccessesList_)
      callerAccesses->mergeWriteOffset(iir::getAccessID(field), getOffset(true));

    for(auto& calleeAccesses : calleeAccessesList_)
      calleeAccesses->mergeWriteOffset(iir::getAccessID(field), getOffset(false));
  }

  void mergeWriteOffset(const std::shared_ptr<iir::VarAccessExpr>& var) {
    for(auto& callerAccesses : callerAccessesList_)
      callerAccesses->mergeWriteOffset(iir::getAccessID(var), ast::Offsets{});

    for(auto& calleeAccesses : calleeAccessesList_)
      calleeAccesses->mergeWriteOffset(iir::getAccessID(var), ast::Offsets{});
  }

  void mergeWriteOffset(const std::shared_ptr<iir::VarDeclStmt>& var) {
    for(auto& callerAccesses : callerAccessesList_)
      callerAccesses->mergeWriteOffset(iir::getAccessID(var), ast::Offsets{});

    for(auto& calleeAccesses : calleeAccessesList_)
      calleeAccesses->mergeWriteOffset(iir::getAccessID(var), ast::Offsets{});
  }

  void mergeWriteExtent(const std::shared_ptr<iir::FieldAccessExpr>& field,
                        const iir::Extents& extent) {
    for(auto& callerAccesses : callerAccessesList_)
      callerAccesses->mergeWriteExtent(iir::getAccessID(field), extent);

    for(auto& calleeAccesses : calleeAccessesList_)
      calleeAccesses->mergeWriteExtent(iir::getAccessID(field), extent);
  }
  /// @}

  /// @brief Add a read offset/extent to the caller and callee accesses
  /// @{
  void mergeReadOffset(const std::shared_ptr<iir::FieldAccessExpr>& field) {
    auto getOffset = [&](bool computeInitialOffset) {
      return (stencilFun_ ? stencilFun_->evalOffsetOfFieldAccessExpr(field, computeInitialOffset)
                          : field->getOffset());
    };

    for(auto& callerAccesses : callerAccessesList_)
      callerAccesses->mergeReadOffset(iir::getAccessID(field), getOffset(true));

    for(auto& calleeAccesses : calleeAccessesList_)
      calleeAccesses->mergeReadOffset(iir::getAccessID(field), getOffset(false));
  }

  void mergeReadOffset(const std::shared_ptr<iir::VarAccessExpr>& var) {
    for(auto& callerAccesses : callerAccessesList_)
      callerAccesses->mergeReadOffset(iir::getAccessID(var), ast::Offsets{});

    for(auto& calleeAccesses : calleeAccessesList_)
      calleeAccesses->mergeReadOffset(iir::getAccessID(var), ast::Offsets{});
  }

  void mergeReadOffset(const std::shared_ptr<iir::LiteralAccessExpr>& lit) {
    for(auto& callerAccesses : callerAccessesList_)
      callerAccesses->mergeReadOffset(iir::getAccessID(lit), ast::Offsets{});

    for(auto& calleeAccesses : calleeAccessesList_)
      calleeAccesses->mergeReadOffset(iir::getAccessID(lit), ast::Offsets{});
  }

  void mergeReadExtent(const std::shared_ptr<iir::FieldAccessExpr>& field,
                       const iir::Extents& extent) {
    for(auto& callerAccesses : callerAccessesList_)
      callerAccesses->mergeReadExtent(iir::getAccessID(field), extent);

    for(auto& calleeAccesses : calleeAccessesList_)
      calleeAccesses->mergeReadExtent(iir::getAccessID(field), extent);
  }
  /// @}

  /// @brief Recursively merge the `extent` with all fields of the `curStencilFunCall` and apply
  /// them to the current *caller* accesses
  template <typename TExtent>
  void
  mergeExtentWithAllFields(TExtent&& extent,
                           std::shared_ptr<iir::StencilFunctionInstantiation> curStencilFunCall,
                           std::set<int>& appliedAccessIDs) {
    for(const iir::Field& field : curStencilFunCall->getCallerFields()) {
      int AccessID = field.getAccessID();

      if(appliedAccessIDs.count(AccessID))
        continue;

      if(curStencilFunCall->isProvidedByStencilFunctionCall(AccessID)) {
        // The field is provided by a stencil function, forward to the callee
        mergeExtentWithAllFields(
            std::forward<TExtent>(extent),
            curStencilFunCall->getFunctionInstantiationOfArgField(
                curStencilFunCall->getArgumentIndexFromCallerAccessID(AccessID)),
            appliedAccessIDs);

      } else {
        appliedAccessIDs.insert(AccessID);

        if(field.getIntend() == iir::Field::IntendKind::Input ||
           field.getIntend() == iir::Field::IntendKind::InputOutput)
          for(auto& callerAccesses : callerAccessesList_)
            callerAccesses->addReadExtent(AccessID, std::forward<TExtent>(extent));

        if(field.getIntend() == iir::Field::IntendKind::Output ||
           field.getIntend() == iir::Field::IntendKind::InputOutput)
          for(auto& callerAccesses : callerAccessesList_)
            callerAccesses->addWriteExtent(AccessID, std::forward<TExtent>(extent));
      }
    }
  }

  virtual void visit(const std::shared_ptr<iir::BlockStmt>& stmt) override {
    appendNewAccesses();
    for(const auto& s : stmt->getStatements()) {

      DAWN_ASSERT(!curStatementStack_.empty());

      curStatementStack_.push_back(std::make_unique<CurrentStatement>(s));

      // Process the statement
      s->accept(*this);

      curStatementStack_.pop_back();
    }
    removeLastChildAccesses();
  }

  virtual void visit(const std::shared_ptr<iir::LoopStmt>& stmt) override {
    stmt->getBlockStmt()->accept(*this);
  }

  virtual void visit(const std::shared_ptr<iir::ExprStmt>& stmt) override {
    appendNewAccesses();
    stmt->getExpr()->accept(*this);
    removeLastChildAccesses();
  }

  virtual void visit(const std::shared_ptr<iir::ReturnStmt>& stmt) override {
    appendNewAccesses();
    stmt->getExpr()->accept(*this);
    removeLastChildAccesses();
  }

  void visit(const std::shared_ptr<iir::IfStmt>& stmt) override {
    appendNewAccesses();

    curStatementStack_.push_back(std::make_unique<CurrentStatement>(stmt->getCondStmt()));
    stmt->getCondStmt()->accept(*this);
    curStatementStack_.pop_back();

    curStatementStack_.back()->IfCondExpr = stmt->getCondExpr();

    curStatementStack_.push_back(std::make_unique<CurrentStatement>(stmt->getThenStmt()));
    stmt->getThenStmt()->accept(*this);
    curStatementStack_.pop_back();
    if(stmt->hasElse()) {
      curStatementStack_.push_back(std::make_unique<CurrentStatement>(stmt->getElseStmt()));
      stmt->getElseStmt()->accept(*this);
      curStatementStack_.pop_back();
    }

    curStatementStack_.back()->IfCondExpr = nullptr;
    removeLastChildAccesses();
  }

  virtual void visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) override {
    appendNewAccesses();

    // Declaration of variables are by defintion writes
    mergeWriteOffset(stmt);

    for(const auto& expr : stmt->getInitList())
      expr->accept(*this);

    removeLastChildAccesses();
  }
  virtual void visit(const std::shared_ptr<iir::ReductionOverNeighborExpr>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
  }

  virtual void visit(const std::shared_ptr<iir::VerticalRegionDeclStmt>& stmt) override {
    DAWN_ASSERT_MSG(0, "VerticalRegionDeclStmt not allowed in this context");
  }
  virtual void visit(const std::shared_ptr<iir::StencilCallDeclStmt>& stmt) override {
    DAWN_ASSERT_MSG(0, "StencilCallDeclStmt not allowed in this context");
  }

  virtual void visit(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt) override {
    DAWN_ASSERT_MSG(0, "BoundaryConditionDeclStmt not allowed in this context");
  }

  virtual void visit(const std::shared_ptr<iir::FunCallExpr>& expr) override {
    for(auto& s : expr->getChildren())
      s->accept(*this);
  }

  virtual void visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) override {

    StencilFunctionCallScope* previousStencilFunCallScope = nullptr;
    if(!stencilFunCalls_.empty()) {

      previousStencilFunCallScope = stencilFunCalls_.top().get();
    }

    // Compute the accesses of the stencil function
    stencilFunCalls_.push(
        std::make_unique<StencilFunctionCallScope>(getStencilFunctionInstantiation(expr)));

    std::shared_ptr<iir::StencilFunctionInstantiation> curStencilFunCall =
        stencilFunCalls_.top()->FunctionInstantiation;
    computeAccesses(curStencilFunCall, curStencilFunCall->getStatements());

    // Compute the fields to get the IOPolicy of the arguments
    curStencilFunCall->update();

    // Traverse the Arguments
    for(const auto& arg : expr->getArguments()) {
      arg->accept(*this);
    }

    // If the current stencil function is called within the argument list of another stencil
    // function, we need to merge the extents
    if(previousStencilFunCallScope) {
      auto& prevArgumentIndex = previousStencilFunCallScope->ArgumentIndex;
      auto& prevStencilFunCall = previousStencilFunCallScope->FunctionInstantiation;

      const iir::Field& field =
          prevStencilFunCall->getCallerFieldFromArgumentIndex(prevArgumentIndex);
      DAWN_ASSERT(prevStencilFunCall->isProvidedByStencilFunctionCall(field.getAccessID()));

      // Add the extent of the field mapping to the stencil function to all fields
      std::set<int> appliedAccessIDs;
      mergeExtentWithAllFields(field.getExtents(), curStencilFunCall, appliedAccessIDs);

      prevArgumentIndex += 1;
    }

    // Done with the current stencil function call
    stencilFunCalls_.pop();
  }

  virtual void visit(const std::shared_ptr<iir::StencilFunArgExpr>& expr) override {
    stencilFunCalls_.top()->ArgumentIndex += 1;
  }

  void visit(const std::shared_ptr<iir::AssignmentExpr>& expr) override {
    // LHS is a write, we resolve this manually as we only care about FieldAccessExpr and
    // VarAccessExpr. However, if we have an expression `a += 5` we need to register the access as
    // write and read!
    bool readAndWrite = expr->getOp() == "+=" || expr->getOp() == "-=" || expr->getOp() == "/=" ||
                        expr->getOp() == "*=" || expr->getOp() == "|=" || expr->getOp() == "&=";

    if(isa<iir::FieldAccessExpr>(expr->getLeft().get())) {
      auto field = std::static_pointer_cast<iir::FieldAccessExpr>(expr->getLeft());
      if(readAndWrite)
        mergeReadOffset(field);

      mergeWriteOffset(field);
    } else if(isa<iir::VarAccessExpr>(expr->getLeft().get())) {
      auto var = std::static_pointer_cast<iir::VarAccessExpr>(expr->getLeft());
      if(readAndWrite)
        mergeReadOffset(var);

      mergeWriteOffset(var);
    }

    // RHS are read accesses
    expr->getRight()->accept(*this);
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

  void visit(const std::shared_ptr<iir::VarAccessExpr>& expr) override {
    // This is always a read access (writes are resolved in handling of the declaration of the
    // variable and in the assignment)
    mergeReadOffset(expr);

    // Resolve the index if this is an array access
    if(expr->isArrayAccess())
      expr->getIndex()->accept(*this);
  }

  void visit(const std::shared_ptr<iir::LiteralAccessExpr>& expr) override {
    // Literals can, by defintion, only be read
    mergeReadOffset(expr);
  }

  void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override {
    if(!stencilFunCalls_.empty()) {

      std::shared_ptr<iir::StencilFunctionInstantiation> functionInstantiation =
          stencilFunCalls_.top()->FunctionInstantiation;
      int& ArgumentIndex = stencilFunCalls_.top()->ArgumentIndex;

      // Get the extent of the field corresponding to the argument index
      const iir::Field& field =
          functionInstantiation->getCallerFieldFromArgumentIndex(ArgumentIndex);

      if(field.getIntend() == iir::Field::IntendKind::Input ||
         field.getIntend() == iir::Field::IntendKind::InputOutput)
        mergeReadExtent(expr, field.getExtents());

      if(field.getIntend() == iir::Field::IntendKind::Output ||
         field.getIntend() == iir::Field::IntendKind::InputOutput)
        mergeWriteExtent(expr, field.getExtents());

      ArgumentIndex += 1;

    } else {
      // This is always a read access (writes are resolved in the the assignment)
      mergeReadOffset(expr);
    }
  }
};

} // anonymous namespace

void computeAccesses(const iir::StencilMetaInformation& metadata,
                     ArrayRef<std::shared_ptr<iir::Stmt>> stmts) {
  for(const auto& stmt : stmts) {
    AccessMapper mapper(metadata, stmt, nullptr);
    stmt->accept(mapper);
  }
}

void computeAccesses(
    std::shared_ptr<iir::StencilFunctionInstantiation> stencilFunctionInstantiation,
    ArrayRef<std::shared_ptr<iir::Stmt>> stmts) {
  for(const auto& stmt : stmts) {
    AccessMapper mapper(stencilFunctionInstantiation->getStencilInstantiation()->getMetaData(),
                        stmt, stencilFunctionInstantiation);
    stmt->accept(mapper);
  }
}

// anonymous namespace

} // namespace dawn
