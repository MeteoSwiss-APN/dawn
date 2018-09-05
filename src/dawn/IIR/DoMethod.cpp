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

#include "dawn/IIR/IIR.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/Stage.h"
#include "dawn/IIR/Accesses.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/SIR/Statement.h"
#include <boost/optional.hpp>
#include "dawn/Support/IndexGenerator.h"

namespace dawn {
namespace iir {

DoMethod::DoMethod(Interval interval)
    : interval_(interval), id_(IndexGenerator::Instance().getIndex()) {}

std::unique_ptr<DoMethod> DoMethod::clone() const {
  auto cloneMS = make_unique<DoMethod>(interval_);

  cloneMS->setID(id_);
  cloneMS->setDependencyGraph(derivedInfo_.dependencyGraph_);

  cloneMS->cloneChildrenFrom(*this);
  return cloneMS;
}

Interval& DoMethod::getInterval() { return interval_; }

const Interval& DoMethod::getInterval() const { return interval_; }

void DoMethod::setDependencyGraph(const std::shared_ptr<DependencyGraphAccesses>& DG) {
  derivedInfo_.dependencyGraph_ = DG;
}

boost::optional<Extents> DoMethod::computeMaximumExtents(const int accessID) const {
  boost::optional<Extents> extents;

  for(auto& stmtAccess : getChildren()) {
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

boost::optional<Interval>
DoMethod::computeEnclosingAccessInterval(const int accessID, const bool mergeWithDoInterval) const {
  boost::optional<Interval> interval;

  boost::optional<Extents>&& extents = computeMaximumExtents(accessID);

  if(extents.is_initialized()) {
    if(mergeWithDoInterval)
      extents->addCenter(2);
    return boost::make_optional(getInterval())->extendInterval(*extents);
  }
  return interval;
}

void DoMethod::setInterval(const Interval& interval) { interval_ = interval; }

const std::shared_ptr<DependencyGraphAccesses>& DoMethod::getDependencyGraph() const {
  return derivedInfo_.dependencyGraph_;
}

class CheckNonNullStatementVisitor : public ASTVisitorForwarding, public NonCopyable {
private:
  bool result_ = false;

public:
  CheckNonNullStatementVisitor() {}
  virtual ~CheckNonNullStatementVisitor() override {}

  bool getResult() const { return result_; }

  virtual void visit(const std::shared_ptr<ExprStmt>& expr) override {
    if(!isa<NOPExpr>(expr->getExpr().get()))
      result_ = true;
    else {
      ASTVisitorForwarding::visit(expr);
    }
  }
};

bool DoMethod::isEmptyOrNullStmt() const {
  for(auto const& statementAccessPair : children_) {
    const std::shared_ptr<Stmt>& root = statementAccessPair->getStatement()->ASTStmt;
    CheckNonNullStatementVisitor checker;
    root->accept(checker);

    if(checker.getResult()) {
      return false;
    }
  }
  return true;
}

} // namespace iir
} // namespace dawn
