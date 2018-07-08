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

#include "dawn/Optimizer/DoMethod.h"
#include "dawn/Optimizer/Accesses.h"
#include "dawn/Optimizer/DependencyGraphAccesses.h"
#include "dawn/Optimizer/StatementAccessesPair.h"
#include "dawn/SIR/Statement.h"
#include <boost/optional.hpp>

namespace dawn {

DoMethod::DoMethod(Stage* stage, Interval interval)
    : stage_(stage), interval_(interval), dependencyGraph_(nullptr) {}

std::vector<std::shared_ptr<StatementAccessesPair>>& DoMethod::getStatementAccessesPairs() {
  return statementAccessesPairs_;
}

const std::vector<std::shared_ptr<StatementAccessesPair>>&
DoMethod::getStatementAccessesPairs() const {
  return statementAccessesPairs_;
}

Interval& DoMethod::getInterval() { return interval_; }

const Interval& DoMethod::getInterval() const { return interval_; }

Stage* DoMethod::getStage() { return stage_; }

void DoMethod::setDependencyGraph(const std::shared_ptr<DependencyGraphAccesses>& DG) {
  dependencyGraph_ = DG;
}

std::shared_ptr<DependencyGraphAccesses>& DoMethod::getDependencyGraph() {
  return dependencyGraph_;
}

boost::optional<Extents> DoMethod::computeMaximumExtents(const int accessID) const {
  boost::optional<Extents> extents;

  for(auto& stmtAccess : getStatementAccessesPairs()) {
    auto extents_ = stmtAccess->computeMaximumExtents(accessID);
    if(!extents_.is_initialized())
      continue;

    if(extents.is_initialized()) {
      extents->merge(*extents_);
    } else {
      extents = extents_;
    }
  }
  return extents;
}

boost::optional<Interval> DoMethod::computeEnclosingAccessInterval(const int accessID) const {
  boost::optional<Interval> interval;

  boost::optional<Extents>&& extents = computeMaximumExtents(accessID);

  if(extents.is_initialized()) {
    auto interv = getInterval();
    return boost::make_optional<Interval>(std::move(interv))->extendInterval(*extents);
  }
  return interval;
}

const std::shared_ptr<DependencyGraphAccesses>& DoMethod::getDependencyGraph() const {
  return dependencyGraph_;
}

class CheckNonNullStatementVisitor : public ASTVisitorForwarding, public NonCopyable {
protected:
  bool result_ = false;

public:
  CheckNonNullStatementVisitor() {}
  virtual ~CheckNonNullStatementVisitor() {}

  bool getResult() const { return result_; }

  virtual void visit(const std::shared_ptr<ExprStmt>& expr) override {
    if(!isa<NOPExpr>(expr->getExpr().get()))
      result_ = true;
  }
};

bool DoMethod::isEmptyOrNullStmt() const {
  std::cout << "OP " << statementAccessesPairs_.size() << std::endl;
  std::cout << statementAccessesPairs_.front()->getStatement()->ASTStmt << std::endl;
  for(auto const& statementAccessPair : statementAccessesPairs_) {
    std::shared_ptr<Stmt> root = statementAccessPair->getStatement()->ASTStmt;
    CheckNonNullStatementVisitor checker;
    root->accept(checker);

    if(checker.getResult())
      return false;
  }
  return true;
}

} // namespace dawn
