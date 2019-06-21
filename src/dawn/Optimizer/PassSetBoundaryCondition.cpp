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

#include "dawn/Optimizer/PassSetBoundaryCondition.h"
#include "dawn/IIR/ControlFlowDescriptor.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/SIR/ASTExpr.h"
#include "dawn/SIR/ASTStmt.h"
#include "dawn/SIR/ASTUtil.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Logging.h"
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>

namespace dawn {

namespace {
/// @brief This method analyzes the Extents in which a field with a given ID is used within a a
/// Stencil. This is used to compute how big the halo exchanges need to be.
/// @param s A shared ptr to the stencil to be analyzed
/// @param ID the FieldID of the Field to be analized
/// @return the full extent of the field in the stencil
static iir::Extents analyzeStencilExtents(const std::unique_ptr<iir::Stencil>& s, int fieldID) {
  iir::Extents fullExtents{0, 0, 0, 0, 0, 0};
  iir::Stencil& stencil = *s;

  int numStages = stencil.getNumStages();

  // loop over stages
  for(int i = 0; i < numStages; ++i) {
    iir::Stage& stage = *(stencil.getStage(i));

    iir::Extents const& stageExtent = stage.getExtents();
    for(const auto& fieldPair : stage.getFields()) {
      const iir::Field& field = fieldPair.second;
      fullExtents.merge(field.getExtents());
      fullExtents.add(stageExtent);
    }
  }

  return fullExtents;
}

enum FieldType { FT_NotOriginal = -1 };
} // namespace
///
/// @brief The VisitStencilCalls class traverses the StencilDescAST to determine an order of the
/// stencil calls. This is required to properly evaluate boundary conditions
///
class VisitStencilCalls : public ASTVisitorForwarding {
  std::vector<std::shared_ptr<StencilCallDeclStmt>> stencilCallsInOrder_;

public:
  const std::vector<std::shared_ptr<StencilCallDeclStmt>>& getStencilCalls() const {
    return stencilCallsInOrder_;
  }
  std::vector<std::shared_ptr<StencilCallDeclStmt>>& getStencilCalls() {
    return stencilCallsInOrder_;
  }

  void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) {
    stencilCallsInOrder_.push_back(stmt);
  }
};

/// @brief The AddBoundaryConditions class traverses the StencilDescAST to extract all the
/// StencilCallStmts for a stencili with a given ID. This is required to properly insert boundary
/// conditions.
class AddBoundaryConditions : public ASTVisitorForwarding {
  std::shared_ptr<iir::StencilInstantiation> instantiation_;
  iir::StencilMetaInformation& metadata_;
  int StencilID_;

  std::vector<std::shared_ptr<StencilCallDeclStmt>> stencilCallsToReplace_;

public:
  AddBoundaryConditions(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                        int StencilID)
      : instantiation_(instantiation), metadata_(instantiation->getMetaData()),
        StencilID_(StencilID) {}

  virtual void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override {
    auto iter = std::find_if(
        metadata_.getStencilCallToStencilIDMap().begin(),
        metadata_.getStencilCallToStencilIDMap().end(),
        [this, stmt](const std::pair<std::shared_ptr<StencilCallDeclStmt>, int>& pair) {
          return pair.first == stmt && pair.second == StencilID_;
        });
    if(iter != metadata_.getStencilCallToStencilIDMap().end()) {
      stencilCallsToReplace_.emplace_back(stmt);
    }
  }

  std::vector<std::shared_ptr<StencilCallDeclStmt>>& getStencilCallsToReplace() {
    return stencilCallsToReplace_;
  }

  void reset() { stencilCallsToReplace_.clear(); }
};

PassSetBoundaryCondition::PassSetBoundaryCondition() : Pass("PassSetBoundaryCondition") {}

bool PassSetBoundaryCondition::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {

  iir::StencilMetaInformation& metadata = stencilInstantiation->getMetaData();

  // check if we need to run this pass
  if(stencilInstantiation->getStencils().size() == 1) {
    if(stencilInstantiation->getOptimizerContext()->getOptions().ReportBoundaryConditions) {
      std::cout << "\nPASS: " << getName() << ": " << stencilInstantiation->getName() << " :";
      std::cout << " No boundary conditions applied\n";
    }
    return true;
  }
  // returns the original ID of a variable
  auto getOriginalID = [&](int ID) {
    // This checks if the field was orignially defined and is not a versioned field, a chached field
    // or something else generated by the optimizer.
    // We have to do this since boundary conditions are only defined for their original field.
    auto checkIfFieldWasOriginallyDefined = [&](int fieldID) {
      return metadata.hasNameToAccessID(stencilInstantiation->getOriginalNameFromAccessID(fieldID));
    };

    // get the original ID by fetching the original name first
    //
    // name:variable --------> ID:1
    //           ^
    //           |---------------|
    //                           |
    // name:optimizer_var       ID:7
    //
    if(checkIfFieldWasOriginallyDefined(ID)) {
      if(metadata.isAccessType(iir::FieldAccessType::FAT_Field, ID)) {
        return metadata.getAccessIDFromName(stencilInstantiation->getOriginalNameFromAccessID(ID));
      } else {
        return (int)FieldType::FT_NotOriginal;
      }
    } else {
      return (int)FieldType::FT_NotOriginal;
    }
  };

  std::unordered_map<int, iir::Extents> dirtyFields;
  std::unordered_map<int, std::shared_ptr<BoundaryConditionDeclStmt>> allBCs;

  //  // Fetch all the boundary conditions stored in the instantiation
  std::transform(metadata.getFieldNameToBCMap().begin(), metadata.getFieldNameToBCMap().end(),
                 std::inserter(allBCs, allBCs.begin()),
                 [&](std::pair<std::string, std::shared_ptr<BoundaryConditionDeclStmt>> bcPair) {
                   return std::make_pair(metadata.getAccessIDFromName(bcPair.first), bcPair.second);
                 });

  // Get the order in which the stencils are called:
  VisitStencilCalls findStencilCalls;

  const iir::ControlFlowDescriptor& controlFlow =
      stencilInstantiation->getIIR()->getControlFlowDescriptor();

  for(const std::shared_ptr<Statement>& statement : controlFlow.getStatements()) {
    statement->ASTStmt->accept(findStencilCalls);
  }
  std::unordered_set<int> StencilIDsVisited_;
  for(const auto& stencilcall : findStencilCalls.getStencilCalls()) {
    int stencilID = metadata.getStencilIDFromStencilCallStmt(stencilcall);
    StencilIDsVisited_.emplace(stencilID);
  }

  auto calculateHaloExtents = [&](std::string fieldname) {
    iir::Extents fullExtent{0, 0, 0, 0, 0, 0};
    // Did we already apply a BoundaryCondition for this field?
    // This is the first time we apply a BC to this field, we traverse all stencils that were
    // applied before
    std::unordered_set<int> stencilIDsToVisit(StencilIDsVisited_);
    if(StencilBCsApplied_.count(fieldname) != 0) {
      for(int traveresedID : StencilBCsApplied_[fieldname]) {
        stencilIDsToVisit.erase(traveresedID);
      }
    }
    for(const auto& stencil : stencilInstantiation->getStencils()) {
      if(stencilIDsToVisit.count(stencil->getStencilID())) {
        fullExtent.merge(analyzeStencilExtents(stencil, metadata.getAccessIDFromName(fieldname)));
        if(StencilBCsApplied_.count(fieldname) == 0) {
          StencilBCsApplied_.emplace(fieldname, std::vector<int>{stencil->getStencilID()});
        } else {
          StencilBCsApplied_[fieldname].push_back(stencil->getStencilID());
        }
      }
    }
    return fullExtent;
  };

  auto insertExtentsIntoMap = [](int fieldID, iir::Extents extents,
                                 std::unordered_map<int, iir::Extents>& map) {
    auto fieldExtentPair = map.find(fieldID);
    if(fieldExtentPair == map.end()) {
      map.emplace(fieldID, extents);
    } else {
      fieldExtentPair->second.merge(extents);
    }
  };

  // Loop through all the StmtAccessPair in the stencil forward
  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    iir::Stencil& stencil = *stencilPtr;
    DAWN_LOG(INFO) << "analyzing stencil " << stencilInstantiation->getName();
    std::unordered_map<int, iir::Extents> stencilDirtyFields;
    stencilDirtyFields.clear();

    for(const auto& stmtAccess : iterateIIROver<iir::StatementAccessesPair>(stencil)) {

      iir::Accesses& acesses = *(stmtAccess->getAccesses());
      const auto& allReadAccesses = acesses.getReadAccesses();
      const auto& allWriteAccesses = acesses.getWriteAccesses();

      // ReadAccesses can trigger Halo-Updates and Boundary conditions if the following
      // criteria are fullfilled:
      // It is a Field (ID!=-1) and we had a write before from another stencil (is in
      // dirtyFields)
      for(const auto& readaccess : allReadAccesses) {
        int originalID = getOriginalID(readaccess.first);
        if(originalID == FieldType::FT_NotOriginal)
          continue;
        if(!dirtyFields.count(originalID))
          continue;
        // If the access is horizontally pointwise we do not need to trigger a BC
        if(readaccess.second.isHorizontalPointwise())
          continue;
        auto IDtoBCpair = allBCs.find(originalID);
        // Check if a boundary condition for this variable was defined
        if(IDtoBCpair == allBCs.end()) {
          DAWN_ASSERT_MSG(
              false, dawn::format("In stencil %s we need a halo update on field %s but no "
                                  "boundary condition is set.\nUpdate the stencil (outside the "
                                  "do-method) with a boundary condition that calls a "
                                  "stencil_function, e.g \n'boundary_condition(zero(), %s);'\n",
                                  stencilInstantiation->getName(),
                                  stencilInstantiation->getOriginalNameFromAccessID(originalID),
                                  stencilInstantiation->getOriginalNameFromAccessID(originalID))
                         .c_str());
        }
        // Calculate the extent and add it to the boundary-condition - Extent map
        iir::Extents fullExtents =
            calculateHaloExtents(metadata.getFieldNameFromAccessID(readaccess.first));
        stencilInstantiation->getMetaData().insertBoundaryConditiontoExtentPair(IDtoBCpair->second,
                                                                                fullExtents);

        auto it = std::find_if(stencilInstantiation->getIIR()->childrenBegin(),
                               stencilInstantiation->getIIR()->childrenEnd(),
                               [&stencil](const std::unique_ptr<iir::Stencil>& storedStencil) {
                                 return storedStencil->getStencilID() == stencil.getStencilID();
                               });
        DAWN_ASSERT_MSG(it != stencilInstantiation->getIIR()->childrenEnd(),
                        "Stencil Triggering the Boundary Condition is not called");

        // Find all the calls to this stencil before which we need to apply the boundary
        // condition. These calls are then replaced by {boundary_condition, stencil_call}
        AddBoundaryConditions visitor(stencilInstantiation, stencil.getStencilID());

        for(auto& statement : controlFlow.getStatements()) {
          visitor.reset();

          std::shared_ptr<Stmt>& stmt = statement->ASTStmt;

          stmt->accept(visitor);
          std::vector<std::shared_ptr<Stmt>> stencilCallWithBC_;
          stencilCallWithBC_.emplace_back(IDtoBCpair->second);
          for(auto& oldStencilCall : visitor.getStencilCallsToReplace()) {
            stencilCallWithBC_.emplace_back(oldStencilCall);
            auto newBlockStmt = std::make_shared<BlockStmt>();
            std::copy(stencilCallWithBC_.begin(), stencilCallWithBC_.end(),
                      std::back_inserter(newBlockStmt->getStatements()));
            if(oldStencilCall == stmt) {
              // Replace the the statement directly
              DAWN_ASSERT(visitor.getStencilCallsToReplace().size() == 1);
              stmt = newBlockStmt;
            } else {
              // Recursively replace the statement
              replaceOldStmtWithNewStmtInStmt(stmt, oldStencilCall, newBlockStmt);
            }
            stencilCallWithBC_.pop_back();
          }
        }

        // The boundary condition is applied, the field is clean again
        dirtyFields.erase(originalID);
        // we add it to a vector for output
        boundaryConditionInserted_.push_back(originalID);
      }
      // Any write-access requires a halo update once is is read off-center therefore we set
      // the fields to modified
      for(const auto& writeaccess : allWriteAccesses) {
        int originalID = getOriginalID(writeaccess.first);
        if(originalID != FieldType::FT_NotOriginal) {
          insertExtentsIntoMap(originalID, writeaccess.second, stencilDirtyFields);
        }
      }
    }
    // Write all the fields set to dirty within this stencil to the global dirty map
    for(const auto& fieldWithExtends : stencilDirtyFields) {
      insertExtentsIntoMap(fieldWithExtends.first, fieldWithExtends.second, dirtyFields);
    }
  }

  for(const auto& doMethod : iterateIIROver<iir::DoMethod>(*(stencilInstantiation->getIIR()))) {
    doMethod->update(iir::NodeUpdateType::levelAndTreeAbove);
  }

  OptimizerContext* context = stencilInstantiation->getOptimizerContext();
  // Output
  if(context->getOptions().ReportBoundaryConditions) {
    std::cout << "\nPASS: " << getName() << ": " << stencilInstantiation->getName() << " :";
    if(boundaryConditionInserted_.size() == 0) {
      std::cout << " No boundary conditions applied\n";
    }
    for(const auto& ID : boundaryConditionInserted_) {
      std::cout << " Boundary Condition for field '"
                << stencilInstantiation->getOriginalNameFromAccessID(ID) << "' inserted"
                << std::endl;
    }
  }

  return true;
}

} // namespace dawn
