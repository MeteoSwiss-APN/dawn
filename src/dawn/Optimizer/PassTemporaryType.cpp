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
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/NodeUpdateType.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/Utility/TemporaryHandeling.h"
#include "dawn/SIR/ASTVisitor.h"
#include <iostream>
#include <memory>
#include <stack>
#include <unordered_map>
#include <unordered_set>

namespace dawn {

namespace {

class StencilFunArgumentDetector : public ASTVisitorForwarding {
//  iir::StencilInstantiation& instantiation_;
    const iir::IIR* iir_;
  int AccessID_;

  int argListNesting_;
  bool usedInStencilFun_;

public:
  StencilFunArgumentDetector(iir::IIR* iir, int AccessID)
      : iir_(iir), AccessID_(AccessID), argListNesting_(0),
        usedInStencilFun_(false) {}

  virtual void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override {
    argListNesting_++;
    ASTVisitorForwarding::visit(expr);
    argListNesting_--;
  }

  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {
    if(argListNesting_ > 0 &&
       iir_->getMetaData()->getAccessIDFromExpr(expr) == AccessID_)
      usedInStencilFun_ = true;
  }

  bool usedInStencilFun() const { return usedInStencilFun_; }
};

/// @brief Check if a field, given by `AccessID` is used as an argument of a stencil-function inside
/// any statement of the `stencil`
/// @returns `true` if field is used as an argument
bool usedAsArgumentInStencilFun(const std::unique_ptr<iir::Stencil>& stencil, int AccessID) {
  StencilFunArgumentDetector visitor(stencil->getIIR(), AccessID);
  stencil->accept(visitor);
  return visitor.usedInStencilFun();
}

/// @brief Representation of a Temporary field or variable
struct Temporary {
  enum TemporaryType { TT_LocalVariable, TT_Field };

  Temporary(Temporary const& other) = default;
  Temporary(Temporary&& other) = default;
  Temporary() = delete;
  Temporary(int accessID, TemporaryType type, const iir::Extents& extent)
      : AccessID(accessID), Type(type), Extent(extent) {}

  int AccessID;                    ///< AccessID of the field or variable
  TemporaryType Type : 1;          ///< Type of the temporary
  iir::Stencil::Lifetime Lifetime; ///< Lifetime of the temporary
  iir::Extents Extent;             ///< Accumulated access of the temporary during its lifetime

  /// @brief Dump the temporary
  void dump(const std::shared_ptr<iir::StencilInstantiation>& instantiation) const {
    std::cout << "Temporary : "
              << instantiation->getIIR()->getMetaData()->getNameFromAccessID(AccessID) << " {"
              << "\n  Type=" << (Type == TT_LocalVariable ? "LocalVariable" : "Field")
              << ",\n  Lifetime=" << Lifetime << ",\n  Extent=" << Extent << "\n}\n";
  }
};

} // anonymous namespace

PassTemporaryType::PassTemporaryType() : Pass("PassTemporaryType", true) {}

bool PassTemporaryType::run(const std::shared_ptr<iir::StencilInstantiation>& instantiation) {

  report_.clear();
  std::unordered_map<int, Temporary> temporaries;
  std::unordered_set<int> AccessIDs;

  // Fix temporaries which span over multiple stencils and promote them to 3D allocated fields

  // Fix the temporaries within the same stencil
  for(const auto& stencilPtr : instantiation->getIIR()->getChildren() /* getStencils()*/) {
    temporaries.clear();
    AccessIDs.clear();

    // Loop over all accesses
    for(const auto& statementAccessesPair :
        iterateIIROver<iir::StatementAccessesPair>(*stencilPtr)) {
      auto processAccessMap = [&](const std::unordered_map<int, iir::Extents>& accessMap) {
        for(const auto& AccessIDExtentPair : accessMap) {
          int AccessID = AccessIDExtentPair.first;
          const iir::Extents& extent = AccessIDExtentPair.second;

          // Is it a temporary?
          bool isTemporaryField =
              instantiation->getIIR()->getMetaData()->isTemporaryField(AccessID);
          if(isTemporaryField ||
             (!instantiation->getIIR()->getMetaData()->isGlobalVariable(AccessID) &&
              instantiation->getIIR()->getMetaData()->isVariable(AccessID))) {

            auto it = temporaries.find(AccessID);
            if(it != temporaries.end()) {
              // If we already registered it, update the extent
              it->second.Extent.merge(extent);
            } else {
              // Register the temporary
              AccessIDs.insert(AccessID);
              temporaries.emplace(AccessID, Temporary(AccessID, isTemporaryField
                                                                    ? Temporary::TT_Field
                                                                    : Temporary::TT_LocalVariable,
                                                      extent));
            }
          }
        }
      };

      processAccessMap(statementAccessesPair->getAccesses()->getWriteAccesses());
      processAccessMap(statementAccessesPair->getAccesses()->getReadAccesses());
    }

    auto LifetimeMap = stencilPtr->getLifetime(AccessIDs);
    std::for_each(LifetimeMap.begin(), LifetimeMap.end(),
                  [&](const std::pair<int, iir::Stencil::Lifetime>& lifetimePair) {
                    DAWN_ASSERT(temporaries.count(lifetimePair.first));
                    temporaries.at(lifetimePair.first).Lifetime = lifetimePair.second;
                  });

    // Process each temporary
    for(const auto& AccessIDTemporaryPair : temporaries) {
      int AccessID = AccessIDTemporaryPair.first;
      const Temporary& temporary = AccessIDTemporaryPair.second;

      auto report = [&](const char* action) {
        std::cout << "\nPASS: " << getName() << ": "
                  << instantiation->getIIR()->getMetaData()->getName() << ": " << action << ":"
                  << instantiation->getIIR()->getMetaData()->getOriginalNameFromAccessID(AccessID) << std::endl;
      };

      if(temporary.Type == Temporary::TT_LocalVariable) {
        // If the variable is accessed in multiple Do-Methods, we need to promote it to a field!
        if(!temporary.Lifetime.Begin.inSameDoMethod(temporary.Lifetime.End)) {

          if(instantiation->getIIR()->getOptions().ReportPassTemporaryType)
            report("promote");

          report_.push_back(Report{AccessID, TmpActionMod::promote});
          promoteLocalVariableToTemporaryField(instantiation->getIIR().get(), stencilPtr.get(),
                                               AccessID, temporary.Lifetime);
        }
      } else {
        // If the field is only accessed within the same Do-Method, does not have an extent and is
        // not argument to a stencil function, we can demote it to a local variable
        if(temporary.Lifetime.Begin.inSameDoMethod(temporary.Lifetime.End) &&
           temporary.Extent.isPointwise() && !usedAsArgumentInStencilFun(stencilPtr, AccessID)) {

          if(instantiation->getIIR()->getOptions().ReportPassTemporaryType)
            report("demote");

          report_.push_back(Report{AccessID, TmpActionMod::demote});
          demoteTemporaryFieldToLocalVariable(instantiation->getIIR().get(), stencilPtr.get(),
                                              AccessID, temporary.Lifetime);
        }
      }
    }

    if(!report_.empty()) {
      for(const auto& ms : iterateIIROver<iir::MultiStage>(*stencilPtr)) {
        ms->update(iir::NodeUpdateType::levelAndTreeAbove);
      }
    }
  }
  return true;
}

void PassTemporaryType::fixTemporariesSpanningMultipleStencils(
    iir::IIR* iir,
    const std::vector<std::unique_ptr<iir::Stencil>>& stencils) {
  if(stencils.size() <= 1)
    return;

  for(int i = 0; i < stencils.size(); ++i) {
    for(const auto& fieldi : stencils[i]->getFields()) {

      // Is fieldi a temporary?
      if(fieldi.second.IsTemporary) {

        // Is it referenced in another stencil?
        for(int j = i + 1; j < stencils.size(); ++j) {
          for(const auto& fieldj : stencils[j]->getFields()) {

            // Yes and yes ... promote it to a real storage
            if(fieldi.second.field.getAccessID() == fieldj.second.field.getAccessID() &&
               fieldj.second.IsTemporary) {
              promoteTemporaryFieldToAllocatedField(iir,
                  fieldi.second.field.getAccessID());
            }
          }
        }
      }
    }
  }
}

} // namespace dawn
