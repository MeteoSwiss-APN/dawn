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

#include "dawn/AST/ASTStmt.h"
#include "dawn/AST/ASTExpr.h"
#include "dawn/AST/ASTUtil.h"
#include "dawn/AST/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Casting.h"

namespace dawn {
namespace ast {
//===------------------------------------------------------------------------------------------===//
//     BlockStmt
//===------------------------------------------------------------------------------------------===//

BlockStmt::BlockStmt(std::unique_ptr<StmtData> data, SourceLocation loc)
    : Stmt(std::move(data), Kind::BlockStmt, loc) {}
BlockStmt::BlockStmt(std::unique_ptr<StmtData> data,
                     const std::vector<std::shared_ptr<Stmt>>& statements, SourceLocation loc)
    : Stmt(std::move(data), Kind::BlockStmt, loc), statements_(statements) {
#if DAWN_USING_ASSERTS
  for(const auto& s : statements)
    DAWN_ASSERT_MSG((checkSameDataType(*s)),
                    "Trying to insert child Stmt with different data type");
#endif
}

BlockStmt::BlockStmt(const BlockStmt& stmt) : Stmt(stmt) {
  for(auto s : stmt.getStatements())
    statements_.push_back(s->clone());
}

BlockStmt& BlockStmt::operator=(BlockStmt const& stmt) {
  assign(stmt);
  statements_ = stmt.statements_;
  return *this;
}

BlockStmt::~BlockStmt() {}

void BlockStmt::push_back(std::shared_ptr<Stmt>&& stmt) {
  DAWN_ASSERT(stmt);
  DAWN_ASSERT_MSG((checkSameDataType(*stmt)),
                  "Trying to insert child Stmt with different data type");
  statements_.push_back(stmt);
}

void BlockStmt::substitute(StmtConstIterator position, std::shared_ptr<Stmt>&& replacement) {
  DAWN_ASSERT(replacement);
  DAWN_ASSERT(position >= statements_.cbegin() && position < statements_.cend());
  DAWN_ASSERT_MSG((checkSameDataType(*replacement)),
                  "Trying to insert child Stmt with different data type");
  statements_[StatementList::size_type(position - statements_.cbegin())] = replacement;
}

std::shared_ptr<Stmt> BlockStmt::clone() const { return std::make_shared<BlockStmt>(*this); }

bool BlockStmt::equals(const Stmt* other, bool compareData) const {
  const BlockStmt* otherPtr = dyn_cast<BlockStmt>(other);
  return otherPtr && Stmt::equals(other, compareData) &&
         otherPtr->statements_.size() == statements_.size() &&
         std::equal(statements_.begin(), statements_.end(), otherPtr->statements_.begin(),
                    [compareData](const std::shared_ptr<Stmt>& a, const std::shared_ptr<Stmt>& b) {
                      return a->equals(b.get(), compareData);
                    });
}

void BlockStmt::replaceChildren(std::shared_ptr<Stmt> const& oldStmt,
                                std::shared_ptr<Stmt> const& newStmt) {
  DAWN_ASSERT_MSG((checkSameDataType(*newStmt)),
                  "Trying to insert child Stmt with different data type");
  [[maybe_unused]] bool success = ASTHelper::replaceOperands(oldStmt, newStmt, statements_);
  DAWN_ASSERT_MSG((success), ("Expression not found"));
}

//===------------------------------------------------------------------------------------------===//
//     ExprStmt
//===------------------------------------------------------------------------------------------===//

ExprStmt::ExprStmt(std::unique_ptr<StmtData> data, const std::shared_ptr<Expr>& expr,
                   SourceLocation loc)
    : Stmt(std::move(data), Kind::ExprStmt, loc), expr_(expr) {}

ExprStmt::ExprStmt(const ExprStmt& stmt) : Stmt(stmt), expr_(stmt.getExpr()->clone()) {}

ExprStmt& ExprStmt::operator=(ExprStmt stmt) {
  assign(stmt);
  expr_ = stmt.getExpr();
  return *this;
}

ExprStmt::~ExprStmt() {}

std::shared_ptr<Stmt> ExprStmt::clone() const { return std::make_shared<ExprStmt>(*this); }

bool ExprStmt::equals(const Stmt* other, bool compareData) const {
  const ExprStmt* otherPtr = dyn_cast<ExprStmt>(other);
  return otherPtr && Stmt::equals(other, compareData) &&
         expr_->equals(otherPtr->expr_.get(), compareData);
}

void ExprStmt::replaceChildren(std::shared_ptr<Expr> const& oldExpr,
                               std::shared_ptr<Expr> const& newExpr) {
  DAWN_ASSERT_MSG((oldExpr == expr_ && oldExpr && newExpr), ("Expression not found"));

  expr_ = newExpr;
}

//===------------------------------------------------------------------------------------------===//
//     ReturnStmt
//===------------------------------------------------------------------------------------------===//

ReturnStmt::ReturnStmt(std::unique_ptr<StmtData> data, const std::shared_ptr<Expr>& expr,
                       SourceLocation loc)
    : Stmt(std::move(data), Kind::ReturnStmt, loc), expr_(expr) {}

ReturnStmt::ReturnStmt(const ReturnStmt& stmt) : Stmt(stmt), expr_(stmt.getExpr()->clone()) {}

ReturnStmt& ReturnStmt::operator=(ReturnStmt stmt) {
  assign(stmt);
  expr_ = stmt.getExpr();
  return *this;
}

ReturnStmt::~ReturnStmt() {}

std::shared_ptr<Stmt> ReturnStmt::clone() const { return std::make_shared<ReturnStmt>(*this); }

bool ReturnStmt::equals(const Stmt* other, bool compareData) const {
  const ReturnStmt* otherPtr = dyn_cast<ReturnStmt>(other);
  return otherPtr && Stmt::equals(other, compareData) &&
         expr_->equals(otherPtr->expr_.get(), compareData);
}

void ReturnStmt::replaceChildren(std::shared_ptr<Expr> const& oldExpr,
                                 std::shared_ptr<Expr> const& newExpr) {
  DAWN_ASSERT_MSG((oldExpr == expr_), ("Expression not found"));
  expr_ = newExpr;
}

//===------------------------------------------------------------------------------------------===//
//     VarDeclStmt
//===------------------------------------------------------------------------------------------===//

VarDeclStmt::VarDeclStmt(std::unique_ptr<StmtData> data, const Type& type, const std::string& name,
                         int dimension, const std::string& op, InitList initList,
                         SourceLocation loc)
    : Stmt(std::move(data), Kind::VarDeclStmt, loc), type_(type), name_(name),
      dimension_(dimension), op_(op), initList_(std::move(initList)) {}

VarDeclStmt::VarDeclStmt(const VarDeclStmt& stmt)
    : Stmt(stmt), type_(stmt.getType()), name_(stmt.getName()), dimension_(stmt.getDimension()),
      op_(stmt.getOp()) {
  for(const auto& expr : stmt.getInitList())
    initList_.push_back(expr->clone());
}

VarDeclStmt& VarDeclStmt::operator=(VarDeclStmt stmt) {
  assign(stmt);
  type_ = std::move(stmt.getType());
  name_ = std::move(stmt.getName());
  dimension_ = stmt.getDimension();
  op_ = stmt.getOp();
  initList_ = std::move(stmt.getInitList());
  return *this;
}

VarDeclStmt::~VarDeclStmt() {}

std::shared_ptr<Stmt> VarDeclStmt::clone() const { return std::make_shared<VarDeclStmt>(*this); }

bool VarDeclStmt::equals(const Stmt* other, bool compareData) const {
  const VarDeclStmt* otherPtr = dyn_cast<VarDeclStmt>(other);
  return otherPtr && Stmt::equals(other, compareData) && type_ == otherPtr->type_ &&
         name_ == otherPtr->name_ && dimension_ == otherPtr->dimension_ && op_ == otherPtr->op_ &&
         initList_.size() == otherPtr->initList_.size() &&
         std::equal(initList_.begin(), initList_.end(), otherPtr->initList_.begin(),
                    [compareData](const std::shared_ptr<Expr>& a, const std::shared_ptr<Expr>& b) {
                      return a->equals(b.get(), compareData);
                    });
}

void VarDeclStmt::replaceChildren(std::shared_ptr<Expr> const& oldExpr,
                                  std::shared_ptr<Expr> const& newExpr) {
  [[maybe_unused]] bool success = ASTHelper::replaceOperands(oldExpr, newExpr, initList_);
  DAWN_ASSERT_MSG(success, ("Expression not found"));
}

//===------------------------------------------------------------------------------------------===//
//     VerticalRegionDeclStmt
//===------------------------------------------------------------------------------------------===//

VerticalRegionDeclStmt::VerticalRegionDeclStmt(
    std::unique_ptr<StmtData> data, const std::shared_ptr<sir::VerticalRegion>& verticalRegion,
    SourceLocation loc)
    : Stmt(std::move(data), Kind::VerticalRegionDeclStmt, loc), verticalRegion_(verticalRegion) {
  DAWN_ASSERT_MSG((checkSameDataType(*verticalRegion_->Ast->getRoot())),
                  "Trying to insert vertical region with different data type");
}

VerticalRegionDeclStmt::VerticalRegionDeclStmt(const VerticalRegionDeclStmt& stmt)
    : Stmt(stmt), verticalRegion_(stmt.getVerticalRegion()->clone()) {}

VerticalRegionDeclStmt& VerticalRegionDeclStmt::operator=(VerticalRegionDeclStmt stmt) {
  assign(stmt);
  verticalRegion_ = std::move(stmt.getVerticalRegion());
  return *this;
}

VerticalRegionDeclStmt::~VerticalRegionDeclStmt() {}

std::shared_ptr<Stmt> VerticalRegionDeclStmt::clone() const {
  return std::make_shared<VerticalRegionDeclStmt>(*this);
}

bool VerticalRegionDeclStmt::equals(const Stmt* other, bool compareData) const {
  const VerticalRegionDeclStmt* otherPtr = dyn_cast<VerticalRegionDeclStmt>(other);
  return otherPtr && Stmt::equals(other, compareData) &&
         *(verticalRegion_.get()) == *(otherPtr->verticalRegion_.get());
}

std::shared_ptr<StencilCall> StencilCall::clone() const {
  auto call = std::make_shared<StencilCall>(Callee, Loc);
  call->Args = Args;
  return call;
}

bool StencilCall::operator==(const StencilCall& rhs) const { return bool(this->comparison(rhs)); }

CompareResult StencilCall::comparison(const StencilCall& rhs) const {
  std::string output;
  if(Callee != rhs.Callee) {
    output += dawn::format("[StencilCall mismatch] Callees do not match\n"
                           "  Actual:\n"
                           "    %s\n"
                           "  Expected:\n"
                           "    %s",
                           Callee, rhs.Callee);
    return CompareResult{output, false};
  }
  for(int i = 0; i < Args.size(); ++i) {
    if(Args[i] != rhs.Args[i]) {
      output += "[StencilCall mismatch] Arguments do not match\n";
      output += dawn::format("Names do not match\n"
                             "  Actual:\n"
                             "    %s\n"
                             "  Expected:\n"
                             "    %s",
                             Args[i], rhs.Args[i]);
      return CompareResult{output, false};
    }
  }
  return CompareResult{output, true};
}

//===------------------------------------------------------------------------------------------===//
//     StencilCallDeclStmt
//===------------------------------------------------------------------------------------------===//

StencilCallDeclStmt::StencilCallDeclStmt(std::unique_ptr<StmtData> data,
                                         const std::shared_ptr<StencilCall>& stencilCall,
                                         SourceLocation loc)
    : Stmt(std::move(data), Kind::StencilCallDeclStmt, loc), stencilCall_(stencilCall) {}

StencilCallDeclStmt::StencilCallDeclStmt(const StencilCallDeclStmt& stmt)
    : Stmt(stmt), stencilCall_(stmt.getStencilCall()->clone()) {}

StencilCallDeclStmt& StencilCallDeclStmt::operator=(StencilCallDeclStmt stmt) {
  assign(stmt);
  stencilCall_ = std::move(stmt.getStencilCall());
  return *this;
}

StencilCallDeclStmt::~StencilCallDeclStmt() {}

std::shared_ptr<Stmt> StencilCallDeclStmt::clone() const {
  return std::make_shared<StencilCallDeclStmt>(*this);
}

bool StencilCallDeclStmt::equals(const Stmt* other, bool compareData) const {
  const StencilCallDeclStmt* otherPtr = dyn_cast<StencilCallDeclStmt>(other);
  if(otherPtr) {
    if(stencilCall_) {
      auto res = stencilCall_->comparison(*otherPtr->stencilCall_);
      if(!bool(res)) {
        return false;
      }
    }
    return Stmt::equals(other, compareData);
  }
  return false;
}

//===------------------------------------------------------------------------------------------===//
//     BoundaryConditionDeclStmt
//===------------------------------------------------------------------------------------------===//

BoundaryConditionDeclStmt::BoundaryConditionDeclStmt(std::unique_ptr<StmtData> data,
                                                     const std::string& callee, SourceLocation loc)
    : Stmt(std::move(data), Kind::BoundaryConditionDeclStmt, loc), functor_(callee) {}

BoundaryConditionDeclStmt::BoundaryConditionDeclStmt(const BoundaryConditionDeclStmt& stmt)
    : Stmt(stmt), functor_(stmt.functor_), fields_(stmt.fields_) {}

BoundaryConditionDeclStmt& BoundaryConditionDeclStmt::operator=(BoundaryConditionDeclStmt stmt) {
  assign(stmt);
  functor_ = std::move(stmt.functor_);
  fields_ = std::move(stmt.fields_);
  return *this;
}

BoundaryConditionDeclStmt::~BoundaryConditionDeclStmt() {}

std::shared_ptr<Stmt> BoundaryConditionDeclStmt::clone() const {
  return std::make_shared<BoundaryConditionDeclStmt>(*this);
}

bool BoundaryConditionDeclStmt::equals(const Stmt* other, bool compareData) const {
  const BoundaryConditionDeclStmt* otherPtr = dyn_cast<BoundaryConditionDeclStmt>(other);
  return otherPtr && Stmt::equals(other, compareData) && functor_ == otherPtr->functor_ &&
         fields_.size() == otherPtr->fields_.size() &&
         std::equal(fields_.begin(), fields_.end(), otherPtr->fields_.begin(),
                    [](const std::string& a, const std::string& b) { return a == b; });
}

//===------------------------------------------------------------------------------------------===//
//     IfStmt
//===------------------------------------------------------------------------------------------===//

IfStmt::IfStmt(std::unique_ptr<StmtData> data, const std::shared_ptr<Stmt>& condStmt,
               const std::shared_ptr<Stmt>& thenStmt, const std::shared_ptr<Stmt>& elseStmt,
               SourceLocation loc)
    : Stmt(std::move(data), Kind::IfStmt, loc), subStmts_{condStmt, thenStmt, elseStmt} {
  for(const auto& s : subStmts_)
    if(s)
      DAWN_ASSERT_MSG((checkSameDataType(*s)), "Trying to insert substmt with different data type");
}

IfStmt::IfStmt(const IfStmt& stmt)
    : Stmt(stmt), subStmts_{stmt.getCondStmt()->clone(), stmt.getThenStmt()->clone(),
                            stmt.hasElse() ? stmt.getElseStmt()->clone() : nullptr} {}

IfStmt& IfStmt::operator=(IfStmt stmt) {
  assign(stmt);
  subStmts_[OperandKind::Cond] = std::move(stmt.getCondStmt());
  subStmts_[OperandKind::Then] = std::move(stmt.getThenStmt());
  subStmts_[OperandKind::Else] = std::move(stmt.getElseStmt());
  return *this;
}

IfStmt::~IfStmt() {}

std::shared_ptr<Stmt> IfStmt::clone() const { return std::make_shared<IfStmt>(*this); }

bool IfStmt::equals(const Stmt* other, bool compareData) const {
  const IfStmt* otherPtr = dyn_cast<IfStmt>(other);
  bool sameElse;
  if(hasElse() && otherPtr->hasElse())
    sameElse = subStmts_[OperandKind::Else]->equals(otherPtr->subStmts_[OperandKind::Else].get(),
                                                    compareData);
  else
    sameElse = !(hasElse() ^ otherPtr->hasElse());
  return otherPtr && Stmt::equals(other, compareData) &&
         subStmts_[OperandKind::Cond]->equals(otherPtr->subStmts_[OperandKind::Cond].get(),
                                              compareData) &&
         subStmts_[OperandKind::Then]->equals(otherPtr->subStmts_[OperandKind::Then].get(),
                                              compareData) &&
         sameElse;
}
void IfStmt::replaceChildren(std::shared_ptr<Stmt> const& oldStmt,
                             std::shared_ptr<Stmt> const& newStmt) {
  DAWN_ASSERT_MSG((checkSameDataType(*newStmt)),
                  "Trying to insert substmt with different data type");
  if(hasElse()) {
    for(std::shared_ptr<Stmt>& stmt : subStmts_) {
      if(stmt == oldStmt)
        stmt = newStmt;
      return;
    }
  } else {
    DAWN_ASSERT(oldStmt == subStmts_[0]);
    subStmts_[0] = newStmt;
    return;
  }
  DAWN_ASSERT_MSG((false), ("Expression not found"));
}

//===------------------------------------------------------------------------------------------===//
//     Loop
//===------------------------------------------------------------------------------------------===//

IterationDescr::~IterationDescr() {}

ChainIterationDescr::ChainIterationDescr(ast::NeighborChain&& chain) : chain_(std::move(chain)) {}
ast::NeighborChain ChainIterationDescr::getChain() const { return chain_; }
std::unique_ptr<IterationDescr> ChainIterationDescr::clone() const {
  return std::make_unique<ChainIterationDescr>(*this);
}
std::string ChainIterationDescr::toString() const {
  std::stringstream ss;
  auto locString = [](ast::LocationType type) {
    switch(type) {
    case ast::LocationType::Cells:
      return "Cell";
    case ast::LocationType::Edges:
      return "Edges";
    case ast::LocationType::Vertices:
      return "Vertices";
    default:
      dawn_unreachable("invalid location type");
    };
  };
  ss << "{";
  bool first = true;
  for(const auto it : chain_) {
    if(!first) {
      ss << ", ";
    }
    first = false;
    ss << locString(it);
  }
  return ss.str();
}
bool ChainIterationDescr::equals(const IterationDescr* otherPtr) const {
  const ChainIterationDescr* chainPtr = dynamic_cast<const ChainIterationDescr*>(otherPtr);
  return chainPtr && chain_ == chainPtr->chain_;
}

LoopStmt::LoopStmt(std::unique_ptr<StmtData> data, ast::NeighborChain&& chain,
                   std::shared_ptr<BlockStmt> body, SourceLocation loc)
    : Stmt(std::move(data), Kind::LoopStmt, loc), blockStmt_(body) {
  iterationDescr_ = std::make_unique<ChainIterationDescr>(std::move(chain));
  for(const auto& stmtPtr : body->getStatements()) {
    DAWN_ASSERT_MSG(stmtPtr->getKind() == ast::Stmt::Kind::ExprStmt,
                    "only expression statements allowed in loop body");
    DAWN_ASSERT_MSG(dyn_pointer_cast<ExprStmt>(stmtPtr)->getExpr()->getKind() ==
                        ast::Expr::Kind::AssignmentExpr,
                    "only assignments allowed in loop body");
  }
}
LoopStmt::LoopStmt(const LoopStmt& stmt)
    : Stmt(stmt), blockStmt_(std::dynamic_pointer_cast<BlockStmt>(stmt.getBlockStmt()->clone())),
      iterationDescr_(stmt.getIterationDescr().clone()) {}
LoopStmt& LoopStmt::operator=(LoopStmt const& stmt) {
  assign(stmt);
  blockStmt_ = stmt.blockStmt_;
  iterationDescr_ = stmt.getIterationDescr().clone();
  return *this;
}
LoopStmt::~LoopStmt() {}

const std::shared_ptr<BlockStmt>& LoopStmt::getBlockStmt() const { return blockStmt_; }
std::shared_ptr<BlockStmt>& LoopStmt::getBlockStmt() { return blockStmt_; }

std::shared_ptr<Stmt> LoopStmt::clone() const { return std::make_shared<LoopStmt>(*this); }
bool LoopStmt::equals(const Stmt* other, bool compareData) const {
  const LoopStmt* otherPtr = dynamic_cast<const LoopStmt*>(other);
  return otherPtr && Stmt::equals(other, compareData) &&
         getBlockStmt()->equals(otherPtr->getBlockStmt().get(), compareData) &&
         iterationDescr_->equals(otherPtr->iterationDescr_.get());
}

MutableArrayRef<std::shared_ptr<Stmt>> LoopStmt::getChildren() { return blockStmt_->getChildren(); }
void LoopStmt::replaceChildren(const std::shared_ptr<Stmt>& oldStmt,
                               const std::shared_ptr<Stmt>& newStmt) {
  blockStmt_->replaceChildren(oldStmt, newStmt);
}

const IterationDescr& LoopStmt::getIterationDescr() const { return *iterationDescr_; }
const IterationDescr* LoopStmt::getIterationDescrPtr() const { return iterationDescr_.get(); }

} // namespace ast
} // namespace dawn
