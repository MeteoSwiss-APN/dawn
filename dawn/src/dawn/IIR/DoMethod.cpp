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

#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/AccessUtils.h"
#include "dawn/IIR/Accesses.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/Stage.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include "dawn/Support/IndexGenerator.h"
#include "dawn/Support/Logging.h"

namespace dawn {
namespace iir {

DoMethod::DoMethod(Interval interval, const StencilMetaInformation& metaData)
    : interval_(interval), id_(IndexGenerator::Instance().getIndex()), metaData_(metaData) {}

std::unique_ptr<DoMethod> DoMethod::clone() const {
  auto cloneMS = std::make_unique<DoMethod>(interval_, metaData_);

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

std::optional<Extents> DoMethod::computeMaximumExtents(const int accessID) const {
  std::optional<Extents> extents;

  for(auto& stmtAccess : getChildren()) {
    auto extents_ = stmtAccess->computeMaximumExtents(accessID);
    if(!extents_)
      continue;

    if(extents) {
      extents->merge(*extents_);
    } else {
      extents = extents_;
    }
  }
  return extents;
}

std::optional<Interval>
DoMethod::computeEnclosingAccessInterval(const int accessID, const bool mergeWithDoInterval) const {
  std::optional<Interval> interval;

  std::optional<Extents>&& extents = computeMaximumExtents(accessID);

  if(extents) {
    if(mergeWithDoInterval)
      extents->addCenter(2);
    return std::make_optional(getInterval())->extendInterval(*extents);
  }
  return interval;
}

void DoMethod::setInterval(const Interval& interval) { interval_ = interval; }

const std::shared_ptr<DependencyGraphAccesses>& DoMethod::getDependencyGraph() const {
  return derivedInfo_.dependencyGraph_;
}

void DoMethod::DerivedInfo::clear() { fields_.clear(); }

void DoMethod::clearDerivedInfo() { derivedInfo_.clear(); }

json::json DoMethod::jsonDump(const StencilMetaInformation& metaData) const {
  json::json node;
  node["ID"] = id_;
  std::stringstream ss;
  ss << interval_;
  node["interval"] = ss.str();

  json::json fieldsJson;
  for(const auto& field : derivedInfo_.fields_) {
    fieldsJson[metaData.getNameFromAccessID(field.first)] = field.second.jsonDump();
  }
  node["Fields"] = fieldsJson;

  json::json stmtsJson;
  for(const auto& stmt : children_) {
    stmtsJson.push_back(stmt->jsonDump(metaData));
  }
  node["Stmts"] = stmtsJson;
  return node;
}

void DoMethod::updateLevel() {

  // Compute the fields and their intended usage. Fields can be in one of three states: `Output`,
  // `InputOutput` or `Input` which implements the following state machine:
  //
  //    +-------+                               +--------+
  //    | Input |                               | Output |
  //    +-------+                               +--------+
  //        |                                       |
  //        |            +-------------+            |
  //        +----------> | InputOutput | <----------+
  //                     +-------------+
  //
  std::unordered_map<int, Field> inputOutputFields;
  std::unordered_map<int, Field> inputFields;
  std::unordered_map<int, Field> outputFields;

  for(const auto& statementAccessesPair : children_) {
    const auto& access = statementAccessesPair->getAccesses();
    DAWN_ASSERT(access);

    for(const auto& accessPair : access->getWriteAccesses()) {
      int AccessID = accessPair.first;
      Extents const& extents = accessPair.second;

      // Does this AccessID correspond to a field access?
      if(!metaData_.isAccessType(FieldAccessType::FAT_Field, AccessID)) {
        continue;
      }
      AccessUtils::recordWriteAccess(inputOutputFields, inputFields, outputFields, AccessID,
                                     extents, getInterval());
    }

    for(const auto& accessPair : access->getReadAccesses()) {
      int AccessID = accessPair.first;
      Extents const& extents = accessPair.second;

      // Does this AccessID correspond to a field access?
      if(!metaData_.isAccessType(FieldAccessType::FAT_Field, AccessID)) {
        continue;
      }

      AccessUtils::recordReadAccess(inputOutputFields, inputFields, outputFields, AccessID, extents,
                                    getInterval());
    }
  }

  // Merge inputFields, outputFields and fields
  derivedInfo_.fields_.insert(outputFields.begin(), outputFields.end());
  derivedInfo_.fields_.insert(inputOutputFields.begin(), inputOutputFields.end());
  derivedInfo_.fields_.insert(inputFields.begin(), inputFields.end());

  if(derivedInfo_.fields_.empty()) {
    DAWN_LOG(WARNING) << "no fields referenced in stage";
    return;
  }

  // Compute the extents of each field by accumulating the extents of each access to field in the
  // stage
  for(const auto& statementAccessesPair : iterateIIROver<StatementAccessesPair>(*this)) {
    const auto& access = statementAccessesPair->getAccesses();

    // first => AccessID, second => Extent
    for(auto& accessPair : access->getWriteAccesses()) {
      if(!metaData_.isAccessType(FieldAccessType::FAT_Field, accessPair.first))
        continue;

      derivedInfo_.fields_.at(accessPair.first).mergeWriteExtents(accessPair.second);
    }

    for(const auto& accessPair : access->getReadAccesses()) {
      if(!metaData_.isAccessType(FieldAccessType::FAT_Field, accessPair.first))
        continue;

      derivedInfo_.fields_.at(accessPair.first).mergeReadExtents(accessPair.second);
    }
  }
}

class CheckNonNullStatementVisitor : public iir::ASTVisitorForwarding, public NonCopyable {
private:
  bool result_ = false;

public:
  CheckNonNullStatementVisitor() {}
  virtual ~CheckNonNullStatementVisitor() override {}

  bool getResult() const { return result_; }

  virtual void visit(const std::shared_ptr<iir::ExprStmt>& expr) override {
    if(!isa<iir::NOPExpr>(expr->getExpr().get()))
      result_ = true;
    else {
      iir::ASTVisitorForwarding::visit(expr);
    }
  }
};

bool DoMethod::isEmptyOrNullStmt() const {
  for(auto const& statementAccessPair : children_) {
    const std::shared_ptr<iir::Stmt>& root = statementAccessPair->getStatement();
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
