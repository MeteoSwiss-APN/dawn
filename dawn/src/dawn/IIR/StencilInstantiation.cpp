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
#include "dawn/AST/ASTStringifier.h"
#include "dawn/AST/LocationType.h"
#include "dawn/IIR/AST.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTUtil.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/Extents.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/InstantiationHelper.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTUtil.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Json.h"
#include "dawn/Support/Logger.h"
#include "dawn/Support/Printing.h"
#include "dawn/Support/RemoveIf.hpp"
#include <cstdlib>
#include <fstream>
#include <functional>
#include <stack>
#include <string>

namespace dawn {
namespace iir {

//===------------------------------------------------------------------------------------------===//
//     StencilInstantiation
//===------------------------------------------------------------------------------------------===//

StencilInstantiation::StencilInstantiation(
    ast::GridType const gridType, std::shared_ptr<sir::GlobalVariableMap> globalVariables,
    std::vector<std::shared_ptr<sir::StencilFunction>> const& stencilFunctions)
    : metadata_(globalVariables),
      IIR_(std::make_unique<IIR>(gridType, globalVariables, stencilFunctions)) {}

StencilMetaInformation& StencilInstantiation::getMetaData() { return metadata_; }

std::shared_ptr<StencilInstantiation> StencilInstantiation::clone() const {

  std::shared_ptr<StencilInstantiation> stencilInstantiation =
      std::make_shared<StencilInstantiation>(IIR_->getGridType(), IIR_->getGlobalVariableMapPtr(),
                                             IIR_->getStencilFunctions());

  stencilInstantiation->metadata_.clone(metadata_);

  stencilInstantiation->IIR_ =
      std::make_unique<iir::IIR>(stencilInstantiation->getIIR()->getGridType(),
                                 stencilInstantiation->getIIR()->getGlobalVariableMapPtr(),
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

const sir::Global& StencilInstantiation::getGlobalVariableValue(const std::string& name) const {
  DAWN_ASSERT(IIR_->getGlobalVariableMap().count(name));
  return IIR_->getGlobalVariableMap().at(name);
}

bool StencilInstantiation::isIDAccessedMultipleStencils(int accessID) const {
  bool wasAccessed = false;
  for(const auto& stencil : IIR_->getChildren()) {
    if(stencil->hasFieldAccessID(accessID)) {
      if(wasAccessed)
        return true;
      wasAccessed = true;
    }
  }
  return false;
}

bool StencilInstantiation::isIDAccessedMultipleMSs(int accessID) const {
  bool wasAccessed = false;
  for(const auto& ms : iterateIIROver<MultiStage>(*IIR_)) {
    if(ms->hasField(accessID)) {
      if(wasAccessed)
        return true;
      wasAccessed = true;
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
  const int AccessID_;
  const bool captureLocation_;

  std::string name_;
  std::vector<SourceLocation> locations_;

public:
  OriginalNameGetter(int AccessID, bool captureLocation)
      : AccessID_(AccessID), captureLocation_(captureLocation) {}

  virtual void visit(const std::shared_ptr<iir::VarDeclStmt>& stmt) override {
    if(iir::getAccessID(stmt) == AccessID_) {
      name_ = stmt->getName();
      if(captureLocation_)
        locations_.push_back(stmt->getSourceLocation());
    }

    for(const auto& expr : stmt->getInitList())
      expr->accept(*this);
  }

  void visit(const std::shared_ptr<iir::VarAccessExpr>& expr) override {
    if(iir::getAccessID(expr) == AccessID_) {
      name_ = expr->getName();
      if(captureLocation_)
        locations_.push_back(expr->getSourceLocation());
    }
  }

  void visit(const std::shared_ptr<iir::LiteralAccessExpr>& expr) override {
    if(iir::getAccessID(expr) == AccessID_) {
      name_ = expr->getValue();
      if(captureLocation_)
        locations_.push_back(expr->getSourceLocation());
    }
  }

  virtual void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override {
    if(iir::getAccessID(expr) == AccessID_) {
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
  OriginalNameGetter orignalNameGetter(AccessID, true);
  stmt->accept(orignalNameGetter);
  return orignalNameGetter.getNameLocationPair();
}

std::string StencilInstantiation::getOriginalNameFromAccessID(int AccessID) const {
  OriginalNameGetter orignalNameGetter(AccessID, true);

  for(const auto& stmt : iterateIIROverStmt(*getIIR())) {
    stmt->accept(orignalNameGetter);
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

template <int Level>
struct PrintDescLine {
  PrintDescLine(std::ostream& os, const std::string& name) : os_(os) {
    os_ << MakeIndent<Level>::value << format("\033[1;3%im", Level) << name << "\n"
        << MakeIndent<Level>::value << "{\n\033[0m";
  }
  ~PrintDescLine() { os_ << MakeIndent<Level>::value << format("\033[1;3%im}\n\033[0m", Level); }

  std::ostream& os_;
};

void StencilInstantiation::dump(std::ostream& os) const {
  os << "StencilInstantiation : " << getName() << "\n";

  int i = 0;
  for(const auto& stencil : getStencils()) {
    PrintDescLine<1> iline(os, "Stencil_" + std::to_string(i));

    int j = 0;
    const auto& multiStages = stencil->getChildren();
    for(const auto& multiStage : multiStages) {
      PrintDescLine<2> jline(os, "MultiStage_" + std::to_string(j) + " [" +
                                     loopOrderToString(multiStage->getLoopOrder()) + "]");

      int k = 0;
      const auto& stages = multiStage->getChildren();
      for(const auto& stage : stages) {
        auto iterSpace = stage->getIterationSpace();
        std::string globidx;
        if(iterSpace[0]) {
          globidx += "I: " + iterSpace[0]->toString() + " ";
        }
        if(iterSpace[1]) {
          globidx += "J: " + iterSpace[1]->toString() + " ";
        }

        PrintDescLine<3> kline(os, "Stage_" + std::to_string(k) + " " + globidx);

        int l = 0;
        const auto& doMethods = stage->getChildren();
        for(const auto& doMethod : doMethods) {
          PrintDescLine<4> lline(os, "Do_" + std::to_string(l) + " " +
                                         std::string(doMethod->getInterval()));

          const auto& stmts = doMethod->getAST().getStatements();
          for(std::size_t m = 0; m < stmts.size(); ++m) {
            os << "\033[1m" << ast::ASTStringifier::toString(stmts[m], 5 * DAWN_PRINT_INDENT)
               << "\033[0m";
            os << stmts[m]->getData<IIRStmtData>().CallerAccesses->toString(
                      [&](int AccessID) { return getMetaData().getNameFromAccessID(AccessID); },
                      6 * DAWN_PRINT_INDENT)
               << "\n";
          }
          l += 1;
        }
        os << "\033[1m" << std::string(4 * DAWN_PRINT_INDENT, ' ')
           << "Extents: " << stage->getExtents() << "\n"
           << "\033[0m";
        k += 1;
      }
      j += 1;
    }
    ++i;
  }
}

void StencilInstantiation::reportAccesses(std::ostream& os) const {
  // Stencil functions
  for(const auto& stencilFun : metadata_.getStencilFunctionInstantiations()) {
    const auto& stmts = stencilFun->getStatements();

    for(std::size_t i = 0; i < stmts.size(); ++i) {
      os << "\nACCESSES: line " << stmts[i]->getSourceLocation().Line << ": "
         << stmts[i]->getData<iir::IIRStmtData>().CalleeAccesses->reportAccesses(stencilFun.get())
         << "\n";
    }
  }

  // Stages
  for(const auto& stmt : iterateIIROverStmt(*getIIR())) {
    os << "\nACCESSES: line " << stmt->getSourceLocation().Line << ": "
       << stmt->getData<iir::IIRStmtData>().CallerAccesses->reportAccesses(metadata_) << "\n";
  }
}

void StencilInstantiation::computeDerivedInfo() {
  // Update doMethod node types
  for(const auto& doMethod : iterateIIROver<iir::DoMethod>(*(this->getIIR()))) {
    doMethod->update(iir::NodeUpdateType::levelAndTreeAbove);
  }

  // Compute stage extents
  for(const auto& stencilPtr : this->getStencils()) {
    iir::Stencil& stencil = *stencilPtr;

    int numStages = stencil.getNumStages();

    // backward loop over stages
    for(int i = numStages - 1; i >= 0; --i) {
      iir::Stage& fromStage = *(stencil.getStage(i));
      // If the stage has a global iterationspace set, we should never extend it since it is user
      // defined where this computation should happen
      if(std::any_of(fromStage.getIterationSpace().cbegin(), fromStage.getIterationSpace().cend(),
                     [](const auto& p) { return p.has_value(); })) {
        fromStage.setExtents(iir::Extents());
        continue;
      }

      iir::Extents const& stageExtent = fromStage.getExtents();

      // loop over all the input fields read in fromStage
      for(const auto& fromFieldPair : fromStage.getFields()) {

        const iir::Field& fromField = fromFieldPair.second;
        auto&& fromFieldExtents = fromField.getExtents();

        // notice that IO (if read happens before write) would also be a valid pattern
        // to trigger the propagation of the stage extents, however this is not a legal
        // pattern within a stage
        // ===-----------------------------------------------------------------------------------===
        //      Point one [ExtentComputationTODO]
        // ===-----------------------------------------------------------------------------------===

        iir::Extents fieldExtent = fromFieldExtents + stageExtent;

        // check which (previous) stage computes the field (read in fromStage)
        for(int j = i - 1; j >= 0; --j) {
          iir::Stage& toStage = *(stencil.getStage(j));
          // ===---------------------------------------------------------------------------------===
          //      Point two [ExtentComputationTODO]
          // ===---------------------------------------------------------------------------------===
          auto fields = toStage.getFields();
          auto it = std::find_if(fields.begin(), fields.end(),
                                 [&](std::pair<int, iir::Field> const& pair) {
                                   const auto& f = pair.second;
                                   return (f.getIntend() != iir::Field::IntendKind::Input) &&
                                          (f.getAccessID() == fromField.getAccessID());
                                 });
          if(it == fields.end())
            continue;

          // if found, add the (read) extent of the field as an extent of the stage
          iir::Extents ext = toStage.getExtents();
          ext.merge(fieldExtent);
          // this pass is computing the redundant computation in the horizontal, therefore we
          // nullify the vertical component of the stage
          ext.resetVerticalExtent();
          toStage.setExtents(ext);
        }
      }
    }
  }

  for(const auto& MS : iterateIIROver<iir::MultiStage>(*(this->getIIR()))) {
    MS->update(iir::NodeUpdateType::levelAndTreeAbove);
  }
}

} // namespace iir
} // namespace dawn
