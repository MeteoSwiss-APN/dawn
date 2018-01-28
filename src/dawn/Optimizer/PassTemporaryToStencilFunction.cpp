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

#include "dawn/Optimizer/PassTemporaryToStencilFunction.h"
#include "dawn/Optimizer/AccessComputation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/Stencil.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"

namespace dawn {

namespace {

/// @brief Register all referenced AccessIDs
struct AccessIDGetter : public ASTVisitorForwarding {
  const StencilInstantiation& Instantiation;
  std::set<int> AccessIDs;

  AccessIDGetter(const StencilInstantiation& instantiation) : Instantiation(instantiation) {}

  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {
    AccessIDs.insert(Instantiation.getAccessIDFromExpr(expr));
  }
};

template <class T>
struct aggregate_adapter : public T {
  template <class... Args>
  aggregate_adapter(Args&&... args) : T{std::forward<Args>(args)...} {}
};

class TmpAssignment : public ASTVisitorForwardingNonConst, public NonCopyable {
protected:
  std::shared_ptr<StencilInstantiation> instantiation_;
  Interval interval_;
  std::shared_ptr<sir::StencilFunction> tmpFunction_;

  // TODO remove, not used
  std::vector<std::shared_ptr<FieldAccessExpr>> params_;
  std::unordered_set<std::string> insertedFields_;

public:
  TmpAssignment(std::shared_ptr<StencilInstantiation> instantiation, Interval const& interval)
      : instantiation_(instantiation), interval_(interval) {}

  virtual ~TmpAssignment() {}

  /// @name Statement implementation
  /// @{
  //  virtual void visit(const std::shared_ptr<BlockStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<ExprStmt>& stmt) override {
    std::cout << "This is expr " << std::endl;

    stmt->getExpr()->accept(*this);
  }
  //  virtual void visit(const std::shared_ptr<IfStmt>& stmt) override;
  /// @}

  /// @name Expression implementation
  /// @{
  virtual void visit(std::shared_ptr<FieldAccessExpr>& expr) override {
    DAWN_ASSERT(tmpFunction_);
    std::cout << "IIIIIIIIIII" << std::endl;
    if(insertedFields_.count(expr->getName()))
      return;
    insertedFields_.emplace(expr->getName());

    params_.push_back(expr);

    tmpFunction_->Args.push_back(std::make_shared<sir::Field>(expr->getName(), SourceLocation{}));
  }

  /// @name Expression implementation
  /// @{
  virtual void visit(const std::shared_ptr<AssignmentExpr>& expr) override {
    std::cout << "KKK " << std::endl;
    if(isa<FieldAccessExpr>(*(expr->getLeft()))) {
      int accessID = instantiation_->getAccessIDFromExpr(expr->getLeft());

      std::cout << " ACC " << accessID << instantiation_->getNameFromAccessID(accessID)
                << std::endl;
      if(!instantiation_->isTemporaryField(accessID))
        return;

      std::string tmpFieldName = instantiation_->getNameFromAccessID(accessID);
      std::cout << " TTT " << accessID << instantiation_->getNameFromAccessID(accessID)
                << std::endl;

      tmpFunction_ = std::make_shared<sir::StencilFunction>();

      tmpFunction_->Name = tmpFieldName + "_OnTheFly";
      tmpFunction_->Loc = expr->getSourceLocation();
      // TODO cretae a interval->sir::interval converter
      tmpFunction_->Intervals.push_back(std::make_shared<aggregate_adapter<sir::Interval>>(
          interval_.lowerLevel(), interval_.upperLevel(), interval_.lowerOffset(),
          interval_.upperOffset()));

      tmpFunction_->Args.push_back(std::make_shared<sir::Offset>("iOffset"));
      tmpFunction_->Args.push_back(std::make_shared<sir::Offset>("jOffset"));
      tmpFunction_->Args.push_back(std::make_shared<sir::Offset>("kOffset"));

      auto functionExpr = expr->getRight()->clone();

      auto retStmt = std::make_shared<ReturnStmt>(functionExpr);

      std::shared_ptr<BlockStmt> root = std::make_shared<BlockStmt>();
      root->push_back(retStmt);
      std::shared_ptr<AST> ast = std::make_shared<AST>(root);

      functionExpr->accept(*this);
    }
  }
  //  virtual void visit(const std::shared_ptr<FunCallExpr>& expr) override;
  //  virtual void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override = 0;
  /// @}
};

} // anonymous namespace

PassTemporaryToStencilFunction::PassTemporaryToStencilFunction()
    : Pass("PassTemporaryToStencilFunction") {}

bool PassTemporaryToStencilFunction::run(
    std::shared_ptr<StencilInstantiation> stencilInstantiation) {
  OptimizerContext* context = stencilInstantiation->getOptimizerContext();

  std::cout << "RUNNING " << std::endl;
  for(auto& stencilPtr : stencilInstantiation->getStencils()) {
    Stencil& stencil = *stencilPtr;

    // Iterate multi-stages backwards
    int stageIdx = stencil.getNumStages() - 1;
    for(auto multiStage : stencil.getMultiStages()) {

      for(const auto& stagePtr : multiStage->getStages()) {

        for(const auto& doMethodPtr : stagePtr->getDoMethods()) {
          for(const auto& stmtAccessPair : doMethodPtr->getStatementAccessesPairs()) {
            const Statement& stmt = *(stmtAccessPair->getStatement());
            std::cout << "INDO " << std::endl;

            if(stmt.ASTStmt->getKind() != Stmt::SK_ExprStmt)
              continue;
            std::cout << "AFTER EXPR " << std::endl;

            TmpAssignment tmpAssignment(stencilInstantiation, doMethodPtr->getInterval());
            stmt.ASTStmt->accept(tmpAssignment);
          }
        }
        //        for(const Field& field : stagePtr->getFields()) {
        //          // This is caching non-temporary fields
        //          if(!instantiation_->isTemporaryField(field.getAccessID()))
        //            continue;
        //        }
      }
      //      std::shared_ptr<DependencyGraphAccesses> newGraph, oldGraph;
      //      newGraph = std::make_shared<DependencyGraphAccesses>(stencilInstantiation.get());

      //      // Iterate stages bottom -> top
      //      for(auto stageRit = multiStage.getStages().rbegin(),
      //               stageRend = multiStage.getStages().rend();
      //          stageRit != stageRend; ++stageRit) {
      //        Stage& stage = (**stageRit);
      //        DoMethod& doMethod = stage.getSingleDoMethod();

      //        // Iterate statements bottom -> top
      //        for(int stmtIndex = doMethod.getStatementAccessesPairs().size() - 1; stmtIndex >= 0;
      //            --stmtIndex) {
      //          oldGraph = newGraph->clone();

      //          auto& stmtAccessesPair = doMethod.getStatementAccessesPairs()[stmtIndex];
      //          newGraph->insertStatementAccessesPair(stmtAccessesPair);

      //          // Try to resolve race-conditions by using double buffering if necessary
      //          auto rc =
      //              fixRaceCondition(newGraph.get(), stencil, doMethod, loopOrder, stageIdx,
      //              stmtIndex);

      //          if(rc == RCKind::RK_Unresolvable)
      //            // Nothing we can do ... bail out
      //            return false;
      //          else if(rc == RCKind::RK_Fixed) {
      //            // We fixed a race condition (this means some fields have changed and our
      //            current graph
      //            // is invalid)
      //            newGraph = oldGraph;
      //            newGraph->insertStatementAccessesPair(stmtAccessesPair);
      //          }
      //        }
      //      }
      //      stageIdx--;
    }
  }

  //  if(context->getOptions().ReportPassTemporaryToStencilFunction && numRenames_ == 0)
  //    std::cout << "\nPASS: " << getName() << ": " << stencilInstantiation->getName()
  //              << ": no rename\n";
  return true;
}

} // namespace dawn
