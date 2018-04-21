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

#include "dawn/Optimizer/StatementAccessesPair.h"
#include "dawn/Optimizer/Accesses.h"
#include "dawn/Optimizer/StencilFunctionInstantiation.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/ASTStringifier.h"
#include "dawn/SIR/Statement.h"
#include "dawn/Support/Printing.h"
#include <sstream>

namespace dawn {

namespace {

template <class InstantiationType>
static std::string toStringImpl(const StatementAccessesPair* pair,
                                const InstantiationType* instantiation, std::size_t initialIndent) {
  std::stringstream ss;

  std::string initialIndentStr = std::string(initialIndent, ' ');
  std::size_t curIndent = initialIndent + DAWN_PRINT_INDENT;
  std::string curIndentStr = std::string(curIndent, ' ');

  ss << initialIndentStr << "[\n" << curIndentStr << "Statement:\n";
  ss << "\e[1m"
     << ASTStringifer::toString(pair->getStatement()->ASTStmt, curIndent + DAWN_PRINT_INDENT)
     << "\e[0m\n";

  if(pair->getCallerAccesses()) {
    ss << curIndentStr << "CallerAccesses:\n";
    ss << pair->getCallerAccesses()->toString(instantiation, curIndent + DAWN_PRINT_INDENT) << "\n";
  }

  if(pair->getCalleeAccesses()) {
    ss << curIndentStr << "CalleeAccesses:\n";
    ss << pair->getCalleeAccesses()->toString(instantiation, curIndent + DAWN_PRINT_INDENT) << "\n";
  }

  if(!pair->getChildren().empty()) {
    ss << curIndentStr << "Children:\n";
    for(auto& child : pair->getChildren())
      ss << child->toString(instantiation, curIndent);
  }
  ss << initialIndentStr << "]\n";

  return ss.str();
}

} // anonymous namespace

StatementAccessesPair::StatementAccessesPair(const std::shared_ptr<Statement>& statement)
    : statement_(statement), callerAccesses_(nullptr), calleeAccesses_(nullptr) {}

std::shared_ptr<Statement> StatementAccessesPair::getStatement() const { return statement_; }

void StatementAccessesPair::setStatement(const std::shared_ptr<Statement>& statement) {
  statement_ = statement;
}

std::shared_ptr<Accesses> StatementAccessesPair::getAccesses() const { return callerAccesses_; }

void StatementAccessesPair::setAccesses(const std::shared_ptr<Accesses>& accesses) {
  callerAccesses_ = accesses;
}

const std::vector<std::shared_ptr<StatementAccessesPair>>&
StatementAccessesPair::getChildren() const {
  return children_;
}

std::vector<std::shared_ptr<StatementAccessesPair>>& StatementAccessesPair::getChildren() {
  return children_;
}

boost::optional<Extents> StatementAccessesPair::computeMaximumExtents(const int accessID) const {
  boost::optional<Extents> extents;

  if(callerAccesses_->hasReadAccess(accessID) || callerAccesses_->hasWriteAccess(accessID)) {
    extents = boost::make_optional(Extents{});
    extents->merge(callerAccesses_->getReadAccess(accessID));
    extents->merge(callerAccesses_->getWriteAccess(accessID));
  }

  for(auto const& child : children_) {
    auto childExtent = child->computeMaximumExtents(accessID);
    if(!childExtent.is_initialized())
      continue;
    if(extents.is_initialized())
      extents->merge(*childExtent);
    else
      extents = childExtent;
  }

  return extents;
}

bool StatementAccessesPair::hasChildren() const { return !children_.empty(); }

std::shared_ptr<Accesses> StatementAccessesPair::getCallerAccesses() const { return getAccesses(); }

void StatementAccessesPair::setCallerAccesses(const std::shared_ptr<Accesses>& accesses) {
  return setAccesses(accesses);
}

std::shared_ptr<Accesses> StatementAccessesPair::getCalleeAccesses() const {
  return calleeAccesses_;
}

void StatementAccessesPair::setCalleeAccesses(const std::shared_ptr<Accesses>& accesses) {
  calleeAccesses_ = accesses;
}

bool StatementAccessesPair::hasCalleeAccesses() { return calleeAccesses_ != nullptr; }

std::string StatementAccessesPair::toString(const StencilInstantiation* instantiation,
                                            std::size_t initialIndent) const {
  return toStringImpl(this, instantiation, initialIndent);
}

std::string StatementAccessesPair::toString(const StencilFunctionInstantiation* stencilFunc,
                                            std::size_t initialIndent) const {
  return toStringImpl(this, stencilFunc, initialIndent);
}

} // namespace dawn
