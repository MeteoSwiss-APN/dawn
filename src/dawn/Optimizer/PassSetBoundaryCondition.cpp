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

/// @brief Get all field and variable accesses identifier by `AccessID`
class GetStencilCalls : public ASTVisitorForwarding {
  StencilInstantiation* instantiation_;
  int StencilID_;

  std::vector<std::shared_ptr<StencilCallDeclStmt>> stencilCallsToReplace_;

public:
  GetStencilCalls(StencilInstantiation* instantiation, int StencilID)
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

struct bcStorage {
  std::string functor;
  std::vector<std::string> arguments;
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
  // returns the original ID of a variable since we need to make sure that if
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
  std::unordered_map<int, bcStorage> allBCs;

  // Fetch all the boundary conditions stored in the instantiation
  for(const auto& bc : stencilInstantiation->getBoundaryConditions()) {
    auto it = stencilInstantiation->getNameToAccessIDMap().find(bc.first);
    if(it != stencilInstantiation->getNameToAccessIDMap().end()) {
      int ID = stencilInstantiation->getAccessIDFromName(bc.first);
      bcStorage storage;
      storage.functor = bc.second.functor;
      for(int i = 0; i < bc.second.arguments.size(); ++i) {
        storage.arguments.push_back(bc.second.arguments[i]);
      }
      allBCs.emplace(ID, storage);
    }
  }

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
              auto idWithExtents = dirtyFields.find(originalID);
              if(originalID != -1 && idWithExtents != dirtyFields.end()) {
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

                    // check if this stencil is called and get its StencilCallDeclStmt (the one to
                    // replace)
                    auto test =
                        stencilInstantiation->getIDToStencilCallMap().find(stencil.getStencilID());
                    if(test != stencilInstantiation->getIDToStencilCallMap().end()) {

                      // Find all the calls to this stencil before which we need to apply the
                      // boundary condition. These calls are then replaced by {boundary_condition,
                      // stencil_call}
                      GetStencilCalls visitor(stencilInstantiation, stencil.getStencilID());

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
                                     stencilInstantiation->getName())
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
