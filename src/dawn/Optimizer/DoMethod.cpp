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

const std::shared_ptr<DependencyGraphAccesses>& DoMethod::getDependencyGraph() const {
  return dependencyGraph_;
}

} // namespace dawn
