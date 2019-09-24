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

#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/InstantiationHelper.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTUtil.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/DiagnosticsEngine.h"
#include "dawn/Support/FileUtil.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Json.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/Printing.h"
#include "dawn/Support/RemoveIf.hpp"
#include "dawn/Support/Twine.h"
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <stack>

namespace dawn {
namespace iir {

//===------------------------------------------------------------------------------------------===//
//     StencilInstantiation
//===------------------------------------------------------------------------------------------===//

StencilInstantiation::StencilInstantiation(
    sir::GlobalVariableMap const& globalVariables,
    std::vector<std::shared_ptr<sir::StencilFunction>> const& stencilFunctions)
    : metadata_(globalVariables), IIR_(make_unique<IIR>(globalVariables, stencilFunctions)) {}

StencilMetaInformation& StencilInstantiation::getMetaData() { return metadata_; }

std::shared_ptr<StencilInstantiation> StencilInstantiation::clone() const {

  std::shared_ptr<StencilInstantiation> stencilInstantiation =
      std::make_shared<StencilInstantiation>(IIR_->getGlobalVariableMap(),
                                             IIR_->getStencilFunctions());

  stencilInstantiation->metadata_.clone(metadata_);

  stencilInstantiation->IIR_ =
      make_unique<iir::IIR>(stencilInstantiation->getIIR()->getGlobalVariableMap(),
                            stencilInstantiation->getIIR()->getStencilFunctions());
  IIR_->clone(stencilInstantiation->IIR_);

  return stencilInstantiation;
}

const std::string StencilInstantiation::getName() const { return metadata_.getStencilName(); }

bool StencilInstantiation::insertBoundaryConditions(
    std::string originalFieldName, std::shared_ptr<iir::BoundaryConditionDeclStmt> bc) {
  if(metadata_.hasFieldBC(originalFieldName) != 0) {
    return false;
  } else {
    metadata_.addFieldBC(originalFieldName, bc);
    return true;
  }
}

const sir::Value& StencilInstantiation::getGlobalVariableValue(const std::string& name) const {
  DAWN_ASSERT(IIR_->getGlobalVariableMap().count(name));
  return *(IIR_->getGlobalVariableMap().at(name));
}

bool StencilInstantiation::isIDAccessedMultipleStencils(int accessID) const {

  int count = 0;
  for(const auto& stencil : IIR_->getChildren()) {
    if(stencil->hasFieldAccessID(accessID)) {
      if(++count > 1)
        return true;
    }
  }
  return false;
}

bool StencilInstantiation::isIDAccessedMultipleMSs(int accessID) const {

  int count = 0;
  for(const auto& ms : iterateIIROver<MultiStage>(*IIR_)) {
    if(ms->hasField(accessID)) {
      if(++count > 1)
        return true;
    }
  }
  return false;
}

std::shared_ptr<StencilFunctionInstantiation>
StencilInstantiation::makeStencilFunctionInstantiation(
    const std::shared_ptr<iir::StencilFunCallExpr>& expr,
    const std::shared_ptr<sir::StencilFunction>& SIRStencilFun,
    const std::shared_ptr<iir::AST>& ast, const Interval& interval,
    const std::shared_ptr<StencilFunctionInstantiation>& curStencilFunctionInstantiation) {

  std::shared_ptr<StencilFunctionInstantiation> stencilFun =
      std::make_shared<StencilFunctionInstantiation>(this, expr, SIRStencilFun, ast, interval,
                                                     curStencilFunctionInstantiation != nullptr);

  metadata_.addStencilFunInstantiationCandidate(
      stencilFun, StencilMetaInformation::StencilFunctionInstantiationCandidate{
                      curStencilFunctionInstantiation});

  return stencilFun;
}

namespace {

/// @brief Get the orignal name of the field (or variable) given by AccessID and a list of
/// SourceLocations where this field (or variable) was accessed.
class OriginalNameGetter : public iir::ASTVisitorForwarding {
  const StencilMetaInformation& metadata_;
  const int AccessID_;
  const bool captureLocation_;

  std::string name_;
  std::vector<SourceLocation> locations_;

public:
  OriginalNameGetter(const StencilMetaInformation& metadata, int AccessID, bool captureLocation)
      : metadata_(metadata), AccessID_(AccessID), captureLocation_(captureLocation) {}

  virtual void visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) override {
    if(metadata_.getAccessIDFromStmt(stmt) == AccessID_) {
      name_ = stmt->getName();
      if(captureLocation_)
        locations_.push_back(stmt->getSourceLocation());
    }

    for(const auto& expr : stmt->getInitList())
      expr->accept(*this);
  }

  void visit(const std::shared_ptr<iir::VarAccessExpr>& expr) override {
    if(metadata_.getAccessIDFromExpr(expr) == AccessID_) {
      name_ = expr->getName();
      if(captureLocation_)
        locations_.push_back(expr->getSourceLocation());
    }
  }

  void visit(const std::shared_ptr<iir::LiteralAccessExpr>& expr) override {
    if(metadata_.getAccessIDFromExpr(expr) == AccessID_) {
      name_ = expr->getValue();
      if(captureLocation_)
        locations_.push_back(expr->getSourceLocation());
    }
  }

  virtual void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override {
    if(metadata_.getAccessIDFromExpr(expr) == AccessID_) {
      name_ = expr->getName();
      if(captureLocation_)
        locations_.push_back(expr->getSourceLocation());
    }
  }

  std::pair<std::string, std::vector<SourceLocation>> getNameLocationPair() const {
    return std::make_pair(name_, locations_);
  }

  bool hasName() const { return !name_.empty(); }
  std::string getName() const { return name_; }
};

} // anonymous namespace

std::pair<std::string, std::vector<SourceLocation>>
StencilInstantiation::getOriginalNameAndLocationsFromAccessID(
    int AccessID, const std::shared_ptr<iir::Stmt>& stmt) const {
  OriginalNameGetter orignalNameGetter(metadata_, AccessID, true);
  stmt->accept(orignalNameGetter);
  return orignalNameGetter.getNameLocationPair();
}

std::string StencilInstantiation::getOriginalNameFromAccessID(int AccessID) const {
  OriginalNameGetter orignalNameGetter(metadata_, AccessID, true);

  for(const auto& stmtAccessesPair : iterateIIROver<StatementAccessesPair>(*getIIR())) {
    stmtAccessesPair->getStatement()->ASTStmt->accept(orignalNameGetter);
    if(orignalNameGetter.hasName())
      return orignalNameGetter.getName();
  }

  // Best we can do...
  return metadata_.getFieldNameFromAccessID(AccessID);
}

bool StencilInstantiation::checkTreeConsistency() const { return IIR_->checkTreeConsistency(); }

void StencilInstantiation::jsonDump(std::string filename) const {

  std::ofstream fs(filename, std::ios::out | std::ios::trunc);
  DAWN_ASSERT(fs.is_open());

  json::json node;
  node["MetaInformation"] = metadata_.jsonDump();
  node["IIR"] = IIR_->jsonDump();
  fs << node.dump(2) << std::endl;
  fs.close();
}

void StencilInstantiation::reportAccesses() const {
  // Stencil functions
  for(const auto& stencilFun : metadata_.getStencilFunctionInstantiations()) {
    const auto& statementAccessesPairs = stencilFun->getStatementAccessesPairs();

    for(std::size_t i = 0; i < statementAccessesPairs.size(); ++i) {
      std::cout << "\nACCESSES: line "
                << statementAccessesPairs[i]->getStatement()->ASTStmt->getSourceLocation().Line
                << ": "
                << statementAccessesPairs[i]->getCalleeAccesses()->reportAccesses(stencilFun.get())
                << "\n";
    }
  }

  // Stages

  for(const auto& stmtAccessesPair : iterateIIROver<StatementAccessesPair>(*getIIR())) {
    std::cout << "\nACCESSES: line "
              << stmtAccessesPair->getStatement()->ASTStmt->getSourceLocation().Line << ": "
              << stmtAccessesPair->getAccesses()->reportAccesses(metadata_) << "\n";
  }
}

} // namespace iir
} // namespace dawn
