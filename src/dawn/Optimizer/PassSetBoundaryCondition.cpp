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
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/Stencil.h"
#include "dawn/Optimizer/StencilInstantiation.h"
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
static dawn::Extents analyzeStencilExtents(const std::shared_ptr<Stencil>& s, int fieldID) {
  Extents fullExtents;
  Stencil& stencil = *s;

  int numStages = stencil.getNumStages();

  // loop over stages
  for(int i = 0; i < numStages; ++i) {
    Stage& stage = *(stencil.getStage(i));

    Extents const& stageExtent = stage.getExtents();
    for(auto& field : stage.getFields()) {
      fullExtents.merge(field.Extent);
      fullExtents.add(stageExtent);
    }
  }

  return fullExtents;
}
}

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
    std::cout << "here: " << stmt->getStencilCall()->Callee << std::endl;
  }
};

/// @brief Get all field and variable accesses identifier by `AccessID`
class AddBoundaryConditions : public ASTVisitorForwarding {
  StencilInstantiation* instantiation_;
  int StencilID_;

  std::vector<std::shared_ptr<StencilCallDeclStmt>> stencilCallsToReplace_;

public:
  AddBoundaryConditions(StencilInstantiation* instantiation, int StencilID)
      : instantiation_(instantiation), StencilID_(StencilID) {}

  void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override {
    if(instantiation_->getStencilIDFromStmt(stmt) == StencilID_)
      stencilCallsToReplace_.emplace_back(stmt);
  }

  std::vector<std::shared_ptr<StencilCallDeclStmt>>& getStencilCallsToReplace() {
    return stencilCallsToReplace_;
  }

  void reset() { stencilCallsToReplace_.clear(); }
};

PassSetBoundaryCondition::PassSetBoundaryCondition() : Pass("PassSetBoundaryCondition") {}

bool PassSetBoundaryCondition::run(StencilInstantiation* stencilInstantiation) {

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
    // check if the variable exists to be sure not to assert
    auto it = stencilInstantiation->getNameToAccessIDMap().find(
        stencilInstantiation->getOriginalNameFromAccessID(ID));
    if(it != stencilInstantiation->getNameToAccessIDMap().end()) {
      if(stencilInstantiation->isField(ID)) {
        return stencilInstantiation->getAccessIDFromName(
            stencilInstantiation->getOriginalNameFromAccessID(ID));
      } else {
        return -1;
      }
    } else {
      return -1;
    }

  };

  std::unordered_map<int, Extents> dirtyFields;
  std::unordered_map<int, BoundaryConditions> allBCs;

  // Fetch all the boundary conditions stored in the instantiation
  for(const auto& bc : stencilInstantiation->getBoundaryConditions()) {
    auto it = stencilInstantiation->getNameToAccessIDMap().find(bc.first);
    if(it != stencilInstantiation->getNameToAccessIDMap().end()) {
      int ID = stencilInstantiation->getAccessIDFromName(bc.first);
      allBCs.emplace(ID, bc.second);
    }
  }

  // Get the order in which the stencils are called:
  VisitStencilCalls findStencilCalls;

  for(const std::shared_ptr<Statement>& statement :
      stencilInstantiation->getStencilDescStatements()) {
    statement->ASTStmt->accept(findStencilCalls);
  }
  std::vector<int> StencilIDsVisited_;
  for(const auto& stencilcall : findStencilCalls.getStencilCalls()) {
    StencilIDsVisited_.push_back(
        stencilInstantiation->getStencilCallToStencilIDMap().find(stencilcall)->second);
  }

  auto calculateHaloExtents = [&](std::string fieldname) {

    Extents fullExtent;
    // Did we already apply a BoundaryCondition for this field?
    // This is the first time we apply a BC to this field, we traverse all stencils that were
    // applied before
    std::vector<int> stencilIDsToVisit(StencilIDsVisited_);
    if(StencilBCsApplied_.count(fieldname) == 0) {
    } else {
      for(int traveresedID : StencilBCsApplied_.find(fieldname)->second) {
        stencilIDsToVisit.erase(
            std::remove(stencilIDsToVisit.begin(), stencilIDsToVisit.end(), traveresedID),
            stencilIDsToVisit.end());
      }
    }
    for(auto& stencil : stencilInstantiation->getStencils()) {
      for(const int& ID : stencilIDsToVisit) {
        if(ID == stencil->getStencilID()) {
          fullExtent.merge(
              analyzeStencilExtents(stencil, stencilInstantiation->getAccessIDFromName(fieldname)));
          if(StencilBCsApplied_.count(fieldname) == 0) {
            StencilBCsApplied_.emplace(fieldname, std::vector<int>{stencil->getStencilID()});
          } else {
            StencilBCsApplied_.find(fieldname)->second.push_back(stencil->getStencilID());
          }
          break;
        }
      }
    }
    return fullExtent;
  };

  // Loop through all the StmtAccessPair in the stencil forward
  for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
    Stencil& stencil = *stencilPtr;
    DAWN_LOG(INFO) << "analyzing stencil " << stencilInstantiation->getName();
    std::unordered_map<int, Extents> stencilDirtyFields;
    stencilDirtyFields.clear();

    for(const auto& multiStagePtr : stencil.getMultiStages()) {
      MultiStage& multiStage = *multiStagePtr;
      for(auto stageIt = multiStage.getStages().begin(); stageIt != multiStage.getStages().end();
          ++stageIt) {
        Stage& stage = (**stageIt);
        for(const auto& domethod : stage.getDoMethods()) {
          for(const auto& stmtAccess : domethod->getStatementAccessesPairs()) {
            Accesses& acesses = *(stmtAccess->getAccesses());
            const auto& allReadAccesses = acesses.getReadAccesses();
            const auto& allWriteAccesses = acesses.getWriteAccesses();

            // ReadAccesses can trigger Halo-Updates and Boundary conditions if the following
            // criteria are fullfilled:
            // It is a Field (ID!=-1) and we had a write before from another stencil (is in
            // dirtyFields)
            for(const auto& readacccess : allReadAccesses) {
              int originalID = getOriginalID(readacccess.first);
              if(originalID == -1)
                continue;
              auto idWithExtents = dirtyFields.find(originalID);
              if(idWithExtents != dirtyFields.end()) {
                // If the access is horizontally pointwise or it is a horizontally cached field,
                // we do not need to trigger a BC
                if(!readacccess.second.isHorizontalPointwise() &&
                   !stencilInstantiation->getCachedVariableSet().count(readacccess.first)) {
                  auto finder = allBCs.find(originalID);
                  // Check if a boundary condition for this variable was defined
                  if(finder != allBCs.end()) {
                    // Create the Statement to insert the Boundary conditon into the StencilDescAST
                    std::shared_ptr<BoundaryConditionDeclStmt> boundaryConditionCall =
                        std::make_shared<BoundaryConditionDeclStmt>(finder->second.functor);
                    boundaryConditionCall->getFields().emplace_back(std::make_shared<sir::Field>(
                        stencilInstantiation->getNameFromAccessID(readacccess.first)));
                    for(const auto& arg : finder->second.arguments) {
                      boundaryConditionCall->getFields().emplace_back(
                          std::make_shared<sir::Field>(arg));
                    }

                    // Calculate the extent and add it to the boundary-condition - Extent map
                    Extents fullExtents = calculateHaloExtents(
                        stencilInstantiation->getNameFromAccessID(readacccess.first));
                    stencilInstantiation->getBoundaryConditionToExtentsMap().emplace(
                        boundaryConditionCall, fullExtents);

                    // check if this stencil is called and get its StencilCallDeclStmt (the one to
                    // replace)
                    auto test =
                        stencilInstantiation->getIDToStencilCallMap().find(stencil.getStencilID());
                    if(test != stencilInstantiation->getIDToStencilCallMap().end()) {

                      // Find all the calls to this stencil before which we need to apply the
                      // boundary condition. These calls are then replaced by {boundary_condition,
                      // stencil_call}
                      AddBoundaryConditions visitor(stencilInstantiation, stencil.getStencilID());

                      for(auto& statement : stencilInstantiation->getStencilDescStatements()) {
                        visitor.reset();

                        std::shared_ptr<Stmt>& stmt = statement->ASTStmt;

                        stmt->accept(visitor);
                        std::vector<std::shared_ptr<Stmt>> stencilCallWithBC_;
                        stencilCallWithBC_.emplace_back(boundaryConditionCall);
                        stencilCallWithBC_.emplace_back(test->second);

                        for(auto& oldStencilCall : visitor.getStencilCallsToReplace()) {
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
                        }
                      }

                      // The boundary condition is applied, the field is clean again
                      dirtyFields.erase(originalID);
                      // we add it to a vector for output
                      boundaryConditionInserted_.push_back(originalID);
                    } else {
                      DAWN_ASSERT_MSG(false,
                                      "Stencil Triggering the Boundary Condition is not called");
                    }
                  } else {
                    DAWN_ASSERT_MSG(
                        false,
                        dawn::format("In stencil %s we need a halo update on field %s but no "
                                     "boundary condition is set.\nUpdate the stencil (outside the "
                                     "do-method) with a boundary condition that calls a "
                                     "stencil_function, e.g \n'boundary_condition(zero(), %s);'\n",
                                     stencilInstantiation->getName(),
                                     stencilInstantiation->getOriginalNameFromAccessID(originalID),
                                     stencilInstantiation->getOriginalNameFromAccessID(originalID))
                            .c_str());
                    dirtyFields.erase(originalID);
                  }
                }
              }
            }
            // Any write-access requires a halo update once is is read off-center therefore we set
            // the fields to modified
            for(const auto& writeaccess : allWriteAccesses) {
              int originalID = getOriginalID(writeaccess.first);
              if(originalID != -1) {
                auto fieldwithExtents = stencilDirtyFields.find(originalID);
                if(fieldwithExtents != stencilDirtyFields.end()) {
                  (*fieldwithExtents).second.merge(writeaccess.second);
                } else {
                  stencilDirtyFields.emplace(originalID, writeaccess.second);
                }
              }
            }
          }
        }
      }
    }
    // Write all the fields set to dirty within this stencil to the global dirty map
    for(const auto& fieldWithExtends : stencilDirtyFields) {
      dirtyFields.emplace(fieldWithExtends.first, fieldWithExtends.second);
    }
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
