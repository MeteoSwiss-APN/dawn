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

#include "dawn/Optimizer/PassTemporaryType.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/NodeUpdateType.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/TemporaryHandling.h"
#include "dawn/Support/Logger.h"
#include <memory>
#include <stack>
#include <unordered_map>
#include <unordered_set>

namespace dawn {

namespace {

class StencilFunArgumentDetector : public iir::ASTVisitorForwarding {
  int AccessID_;

  int argListNesting_;
  bool usedInStencilFun_;

public:
  StencilFunArgumentDetector(int AccessID)
      : AccessID_(AccessID), argListNesting_(0), usedInStencilFun_(false) {}

  virtual void visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) override {
    argListNesting_++;
    iir::ASTVisitorForwarding::visit(expr);
    argListNesting_--;
  }

  virtual void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override {
    if(argListNesting_ > 0 && iir::getAccessID(expr) == AccessID_)
      usedInStencilFun_ = true;
  }

  bool usedInStencilFun() const { return usedInStencilFun_; }
};

/// @brief Check if a field, given by `AccessID` is used as an argument of a stencil-function inside
/// any statement of the `stencil`
/// @returns `true` if field is used as an argument
bool usedAsArgumentInStencilFun(const std::unique_ptr<iir::Stencil>& stencil, int AccessID) {
  StencilFunArgumentDetector visitor(AccessID);
  stencil->accept(visitor);
  return visitor.usedInStencilFun();
}

/// @brief Representation of a Temporary field or variable
struct Temporary {

  Temporary(Temporary const& other) = default;
  Temporary(Temporary&& other) = default;
  Temporary() = delete;
  Temporary(int accessID, iir::TemporaryScope type, const iir::Extents& extent)
      : accessID_(accessID), type_(type), extent_(extent) {}

  int accessID_;                    ///< AccessID of the field or variable
  iir::TemporaryScope type_;        ///< Type of the temporary
  iir::Stencil::Lifetime lifetime_; ///< Lifetime of the temporary
  iir::Extents extent_;             ///< Accumulated access of the temporary during its lifetime

  // TODO remove the dump and should tne lifetime go into the Field as derived info?
  /// @brief Dump the temporary
  void dump(const std::shared_ptr<iir::StencilInstantiation>& instantiation) const {
    DAWN_LOG(INFO) << "Temporary : " << instantiation->getMetaData().getNameFromAccessID(accessID_)
                   << " {"
                   << "\n  Type="
                   << (type_ == iir::TemporaryScope::LocalVariable ? "LocalVariable" : "Field")
                   << ",\n  Lifetime=" << lifetime_ << ",\n  Extent=" << extent_ << "\n}";
  }
};

} // anonymous namespace

PassTemporaryType::PassTemporaryType(OptimizerContext& context)
    : Pass(context, "PassTemporaryType", true) {}

bool PassTemporaryType::run(const std::shared_ptr<iir::StencilInstantiation>& instantiation) {
  const auto& metadata = instantiation->getMetaData();

  report_.clear();
  std::unordered_map<int, Temporary> temporaries;
  std::unordered_set<int> AccessIDs;

  // Fix temporaries which span over multiple stencils and promote them to 3D allocated fields

  // Fix the temporaries within the same stencil
  for(const auto& stencilPtr : instantiation->getStencils()) {
    temporaries.clear();
    AccessIDs.clear();

    // Loop over all accesses
    for(const auto& stmt : iterateIIROverStmt(*stencilPtr)) {
      auto processAccessMap = [&](const std::unordered_map<int, iir::Extents>& accessMap) {
        for(const auto& AccessIDExtentPair : accessMap) {
          int AccessID = AccessIDExtentPair.first;
          const iir::Extents& extent = AccessIDExtentPair.second;

          // Is it a temporary?
          bool isTemporaryField =
              metadata.isAccessType(iir::FieldAccessType::StencilTemporary, AccessID);
          if(isTemporaryField ||
             (!metadata.isAccessType(iir::FieldAccessType::GlobalVariable, AccessID) &&
              metadata.isAccessType(iir::FieldAccessType::LocalVariable, AccessID))) {

            auto it = temporaries.find(AccessID);
            if(it != temporaries.end()) {
              // If we already registered it, update the extent
              it->second.extent_.merge(extent);
            } else {
              // Register the temporary
              AccessIDs.insert(AccessID);
              iir::TemporaryScope ttype = (isTemporaryField ? iir::TemporaryScope::StencilTemporary
                                                            : iir::TemporaryScope::LocalVariable);

              temporaries.emplace(AccessID, Temporary(AccessID, ttype, extent));
            }
          }
        }
      };
      const auto& callerAccesses = stmt->getData<iir::IIRStmtData>().CallerAccesses;
      processAccessMap(callerAccesses->getWriteAccesses());
      processAccessMap(callerAccesses->getReadAccesses());
    }

    auto LifetimeMap = stencilPtr->getLifetime(AccessIDs);
    std::for_each(LifetimeMap.begin(), LifetimeMap.end(),
                  [&](const std::pair<int, iir::Stencil::Lifetime>& lifetimePair) {
                    DAWN_ASSERT(temporaries.count(lifetimePair.first));
                    temporaries.at(lifetimePair.first).lifetime_ = lifetimePair.second;
                  });

    // Process each temporary
    for(const auto& AccessIDTemporaryPair : temporaries) {
      int accessID = AccessIDTemporaryPair.first;
      const Temporary& temporary = AccessIDTemporaryPair.second;

      auto report = [&](const char* action) {
        DAWN_LOG(INFO) << instantiation->getName() << ": " << action << ": "
                       << instantiation->getOriginalNameFromAccessID(accessID);
      };

      // we promote local variables into temporary fields if they are accessed out
      // of a local scope
      if(temporary.type_ == iir::TemporaryScope::LocalVariable) {
        // we promote to a temporary any local variable that is accessed in different Do methods
        // Since the Lifetime only records the position within one stencil we additionally check if
        // the accessID is accessed in multiple stencils
        if(!temporary.lifetime_.Begin.inSameDoMethod(temporary.lifetime_.End) ||
           instantiation->isIDAccessedMultipleMSs(accessID)) {

          if(context_.getOptions().ReportPassTemporaryType)
            report("promote");

          report_.push_back(Report{accessID, TmpActionMod::promote});
          promoteLocalVariableToTemporaryField(instantiation.get(), stencilPtr.get(), accessID,
                                               temporary.lifetime_, temporary.type_);
        }
      } else {
        // If the field is only accessed within the same Do-Method, does not have an extent and is
        // not argument to a stencil function, we can demote it to a local variable
        // Also solver accesses can not be demoted to local variable
        if(temporary.lifetime_.Begin.inSameDoMethod(temporary.lifetime_.End) &&
           temporary.extent_.isPointwise() && !usedAsArgumentInStencilFun(stencilPtr, accessID) &&
           !instantiation->isIDAccessedMultipleMSs(accessID)) {

          if(context_.getOptions().ReportPassTemporaryType)
            report("demote");

          report_.push_back(Report{accessID, TmpActionMod::demote});
          demoteTemporaryFieldToLocalVariable(instantiation.get(), stencilPtr.get(), accessID,
                                              temporary.lifetime_);
        }
      }
    }
    fixTemporariesSpanningMultipleStencils(instantiation.get(), instantiation->getStencils());

    if(!report_.empty()) {
      for(const auto& ms : iterateIIROver<iir::MultiStage>(*stencilPtr)) {
        ms->update(iir::NodeUpdateType::levelAndTreeAbove);
      }
    }
  }
  return true;
} // namespace dawn

void PassTemporaryType::fixTemporariesSpanningMultipleStencils(
    iir::StencilInstantiation* instantiation,
    const std::vector<std::unique_ptr<iir::Stencil>>& stencils) {
  if(stencils.size() <= 1)
    return;

  const auto& metadata = instantiation->getMetaData();
  bool updated = false;
  for(int i = 0; i < stencils.size(); ++i) {
    for(const auto& field : stencils[i]->getFields()) {
      const int accessID = field.first;
      // Is fieldi a temporary?
      // TODO could it happen that the access is not a temporary (but a local var) and even if it is
      // used in multiple stencils there is no need to promote it ?
      if(metadata.isAccessType(iir::FieldAccessType::StencilTemporary, accessID) &&
         instantiation->isIDAccessedMultipleStencils(accessID)) {
        updated = true;
        promoteTemporaryFieldToAllocatedField(instantiation, accessID);
      }
    }
  }
  if(updated) {
    for(const auto& stencil : stencils) {
      stencil->update(iir::NodeUpdateType::level);
    }
  }
}

} // namespace dawn
