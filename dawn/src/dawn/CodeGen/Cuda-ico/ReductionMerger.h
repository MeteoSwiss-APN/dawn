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

#pragma once

#include <cstddef>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <vector>

#include "dawn/AST/ASTExpr.h"
#include "dawn/AST/ASTStmt.h"
#include "dawn/AST/ASTStringifier.h"
#include "dawn/AST/ASTVisitor.h"
#include "dawn/AST/LocationType.h"
#include "dawn/IIR/AST.h"
#include "dawn/IIR/ASTExpr.h"

#include "ASTStencilBody.h"

namespace dawn {
namespace codegen {
namespace cudaico {

class ReductionMergeGroupsComputer {
  class FindMergeGroupsVisitor : public ast::ASTVisitorForwarding {

    std::map<int, std::vector<std::vector<ast::ReductionOverNeighborExpr>>> blockMergeGroups;

    void visit(const std::shared_ptr<const ast::BlockStmt>& stmt) override {
      std::vector<std::vector<ast::ReductionOverNeighborExpr>> mergeGroups;

      int outer_iterator = 0;
      int inner_iterator = 0;

      auto mayContainReduction = [](const ast::Stmt& stmt) -> bool {
        return stmt.getKind() == ast::Stmt::Kind::ExprStmt ||
               stmt.getKind() == ast::Stmt::Kind::VarDeclStmt;
      };

      auto getReduction = [&](ast::Stmt& stmt) -> std::shared_ptr<ast::ReductionOverNeighborExpr> {
        if(!mayContainReduction(stmt)) {
          return 0;
        }
        FindReduceOverNeighborExpr redFinder;
        stmt.accept(redFinder);
        if(!redFinder.hasReduceOverNeighborExpr()) {
          return 0;
        }
        DAWN_ASSERT(redFinder.reduceOverNeighborExprs().size() == 1);
        return redFinder.reduceOverNeighborExprs()[0];
      };

      while(outer_iterator < stmt->getStatements().size()) {
        const auto& curRedcution = getReduction(*stmt->getStatements()[outer_iterator]);
        if(curRedcution) {
          std::vector<ast::ReductionOverNeighborExpr> mergeGroup;
          mergeGroup.push_back(*curRedcution);
          inner_iterator = outer_iterator + 1;
          bool compatible = true;
          while(compatible && inner_iterator < stmt->getStatements().size()) {
            const auto& nextRedcution = getReduction(*stmt->getStatements()[inner_iterator]);
            if(!nextRedcution) {
              compatible = false;
              break;
            } else {
              compatible &= curRedcution->getIterSpace() == nextRedcution->getIterSpace();
              if(compatible) {
                mergeGroup.push_back(*nextRedcution);
                inner_iterator++;
              } else {
                break;
              }
            }
          }
          mergeGroups.push_back(mergeGroup);
          outer_iterator = inner_iterator + 1;
        } else {
          outer_iterator++;
        }
      }
      blockMergeGroups[stmt->getID()] = mergeGroups;
    }

  public:
    std::map<int, std::vector<std::vector<ast::ReductionOverNeighborExpr>>> getMergGroupsByBlock() {
      return blockMergeGroups;
    }

    void dumpMergeGroups() {
      for(const auto& blockIt : blockMergeGroups) {
        std::cout << "BLOCK: " << blockIt.first << "\n";
        for(int groupIt = 0; groupIt < blockIt.second.size(); groupIt++) {
          std::cout << "  GROUP: " << groupIt << "\n";
          for(auto& redIt : blockIt.second[groupIt]) {
            std::cout << "    "
                      << ast::ASTStringifier::toString(
                             std::make_shared<ast::ReductionOverNeighborExpr>(redIt))
                      << "\n";
          }
        }
      }
    }
  };

public:
  static std::map<int, std::vector<std::vector<ast::ReductionOverNeighborExpr>>>
  ComputeReductionMergeGroups(
      const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
    FindMergeGroupsVisitor mergeGroupVtor;
    for(const auto& doMethod : iterateIIROver<iir::DoMethod>(*(stencilInstantiation->getIIR()))) {
      doMethod->getAST().accept(mergeGroupVtor);
    }
    mergeGroupVtor.dumpMergeGroups();
    return mergeGroupVtor.getMergGroupsByBlock();
  }
};

} // namespace cudaico
} // namespace codegen
} // namespace dawn