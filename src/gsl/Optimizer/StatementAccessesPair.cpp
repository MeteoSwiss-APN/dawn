//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gsl/Optimizer/StatementAccessesPair.h"
#include "gsl/Optimizer/Accesses.h"
#include "gsl/Optimizer/StencilFunctionInstantiation.h"
#include "gsl/Optimizer/StencilInstantiation.h"
#include "gsl/SIR/ASTStringifier.h"
#include "gsl/SIR/Statement.h"
#include "gsl/Support/Printing.h"
#include <sstream>

namespace gsl {

namespace {

template <class InstantiationType>
static std::string toStringImpl(const StatementAccessesPair* pair,
                                const InstantiationType* instantiation, std::size_t initialIndent) {
  std::stringstream ss;

  std::string initialIndentStr = std::string(initialIndent, ' ');
  std::size_t curIndent = initialIndent + GSL_PRINT_INDENT;
  std::string curIndentStr = std::string(curIndent, ' ');

  ss << initialIndentStr << "[\n" << curIndentStr << "Statement:\n";
  ss << "\e[1m"
     << ASTStringifer::toString(pair->getStatement()->ASTStmt, curIndent + GSL_PRINT_INDENT)
     << "\e[0m\n";

  if(pair->getCallerAccesses()) {
    ss << curIndentStr << "CallerAccesses:\n";
    ss << pair->getCallerAccesses()->toString(instantiation, curIndent + GSL_PRINT_INDENT) << "\n";
  }

  if(pair->getCalleeAccesses()) {
    ss << curIndentStr << "CalleeAccesses:\n";
    ss << pair->getCalleeAccesses()->toString(instantiation, curIndent + GSL_PRINT_INDENT) << "\n";
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

const std::shared_ptr<Statement>& StatementAccessesPair::getStatement() const { return statement_; }

void StatementAccessesPair::setStatement(const std::shared_ptr<Statement>& statement) {
  statement_ = statement;
}

const std::shared_ptr<Accesses>& StatementAccessesPair::getAccesses() const {
  return callerAccesses_;
}

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

bool StatementAccessesPair::hasChildren() const { return !children_.empty(); }

const std::shared_ptr<Accesses>& StatementAccessesPair::getCallerAccesses() const {
  return getAccesses();
}

void StatementAccessesPair::setCallerAccesses(const std::shared_ptr<Accesses>& accesses) {
  return setAccesses(accesses);
}

const std::shared_ptr<Accesses>& StatementAccessesPair::getCalleeAccesses() const {
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

} // namespace gsl
