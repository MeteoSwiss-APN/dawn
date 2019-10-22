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
#include "dawn/IIR/ASTFwd.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/ASTStringifier.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/AccessToNameMapper.h"
#include "dawn/IIR/AccessUtils.h"
#include "dawn/IIR/Accesses.h"
#include "dawn/IIR/DependencyGraphAccesses.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/Stage.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include "dawn/Support/IndexGenerator.h"
#include "dawn/Support/Logging.h"
#include <limits>
#include <memory>

namespace dawn {
namespace iir {

DoMethod::DoMethod(Interval interval, const StencilMetaInformation& metaData)
    : interval_(interval), id_(IndexGenerator::Instance().getIndex()),
      metaData_(metaData), ast_{std::make_unique<iir::IIRStmtData>()} {}

std::unique_ptr<DoMethod> DoMethod::clone() const {
  auto cloneMS = std::make_unique<DoMethod>(interval_, metaData_);

  cloneMS->setID(id_);
  cloneMS->derivedInfo_ = derivedInfo_.clone();
  cloneMS->ast_ = iir::BlockStmt{ast_};
  return cloneMS;
}

Interval& DoMethod::getInterval() { return interval_; }

const Interval& DoMethod::getInterval() const { return interval_; }

void DoMethod::setDependencyGraph(const std::shared_ptr<DependencyGraphAccesses>& DG) {
  derivedInfo_.dependencyGraph_ = DG;
}

std::optional<Extents> DoMethod::computeMaximumExtents(const int accessID) const {
  std::optional<Extents> extents;

  for(auto& stmt : getChildren()) {
    auto extents_ = iir::computeMaximumExtents(*stmt, accessID);
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
    if(mergeWithDoInterval) {
      extents->addVerticalCenter();
    }
    return std::make_optional(getInterval())->extendInterval(*extents);
  }
  return interval;
}

void DoMethod::setInterval(const Interval& interval) { interval_ = interval; }

const std::shared_ptr<DependencyGraphAccesses>& DoMethod::getDependencyGraph() const {
  return derivedInfo_.dependencyGraph_;
}

DoMethod::DerivedInfo DoMethod::DerivedInfo::clone() const {
  DerivedInfo clone;
  clone.fields_ = fields_;
  clone.dependencyGraph_ = dependencyGraph_ ? dependencyGraph_->clone() : nullptr;
  return clone;
}

void DoMethod::DerivedInfo::clear() { fields_.clear(); }

void DoMethod::clearDerivedInfo() { derivedInfo_.clear(); }

namespace {
json::json print(const StencilMetaInformation& metadata,
                 const AccessToNameMapper& accessToNameMapper,
                 const std::unordered_map<int, Extents>& accesses) {
  json::json node;
  for(const auto& accessPair : accesses) {
    json::json accessNode;
    int accessID = accessPair.first;
    std::string accessName = "unknown";
    if(accessToNameMapper.hasAccessID(accessID)) {
      accessName = accessToNameMapper.getNameFromAccessID(accessID);
    }
    if(metadata.isAccessType(iir::FieldAccessType::FAT_Literal, accessID)) {
      continue;
    }
    accessNode["access_id"] = accessID;
    accessNode["name"] = accessName;
    std::stringstream ss;
    ss << accessPair.second;
    accessNode["extents"] = ss.str();
    node.push_back(accessNode);
  }
  return node;
}
} // namespace

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
  for(const auto& stmt : getChildren()) {
    json::json stmtNode;
    stmtNode["stmt"] = ASTStringifier::toString(stmt, 0);

    AccessToNameMapper accessToNameMapper(metaData);
    stmt->accept(accessToNameMapper);

    stmtNode["write_accesses"] =
        print(metaData, accessToNameMapper,
              stmt->getData<iir::IIRStmtData>().CallerAccesses->getWriteAccesses());
    stmtNode["read_accesses"] =
        print(metaData, accessToNameMapper,
              stmt->getData<iir::IIRStmtData>().CallerAccesses->getReadAccesses());
    stmtsJson.push_back(stmtNode);
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

  for(const auto& stmt : getChildren()) {
    const auto& access = stmt->getData<iir::IIRStmtData>().CallerAccesses;
    DAWN_ASSERT(access);

    for(const auto& accessPair : access->getWriteAccesses()) {
      int AccessID = accessPair.first;
      Extents const& extents = accessPair.second;

      // Does this AccessID correspond to a field access?
      if(!metaData_.isAccessType(FieldAccessType::FAT_Field, AccessID)) {
        continue;
      }

      if(metaData_.getIsUnstructuredFromAccessID(AccessID)) {
        AccessUtils::recordWriteAccess(inputOutputFields, inputFields, outputFields, AccessID,
                                       extents, getInterval(),
                                       metaData_.getLocationTypeFromAccessID(AccessID));
      } else {
        AccessUtils::recordWriteAccess(inputOutputFields, inputFields, outputFields, AccessID,
                                       extents, getInterval());
      }
    }

    for(const auto& accessPair : access->getReadAccesses()) {
      int AccessID = accessPair.first;
      Extents const& extents = accessPair.second;

      // Does this AccessID correspond to a field access?
      if(!metaData_.isAccessType(FieldAccessType::FAT_Field, AccessID)) {
        continue;
      }

      if(metaData_.getIsUnstructuredFromAccessID(AccessID)) {
        AccessUtils::recordReadAccess(inputOutputFields, inputFields, outputFields, AccessID,
                                      extents, getInterval(),
                                      metaData_.getLocationTypeFromAccessID(AccessID));
      } else {
        AccessUtils::recordReadAccess(inputOutputFields, inputFields, outputFields, AccessID,
                                      extents, getInterval());
      }
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
  for(const auto& stmt : getChildren()) {
    const auto& access = stmt->getData<iir::IIRStmtData>().CallerAccesses;

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
  for(auto const& stmt : getChildren()) {
    const std::shared_ptr<iir::Stmt>& root = stmt;
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
