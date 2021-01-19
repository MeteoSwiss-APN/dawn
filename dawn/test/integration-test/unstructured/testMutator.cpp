#include "testMutator.h"
#include "dawn/AST/ASTVisitor.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/AccessComputation.h"
#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/FieldAccessMetadata.h"
#include "dawn/Optimizer/Lowering.h"
#include "dawn/Support/ArrayRef.h"
#include "testMutator.h"

#include <map>
#include <optional>

class accessMutator : public dawn::ast::ASTVisitorForwarding {
  void visit(const std::shared_ptr<dawn::ast::FieldAccessExpr>& expr) override {
    if(mutatedFields_.count(expr->getName())) {
      expr->getOffset().setVerticalIndirection(expr->getName() + "_indirection");
      expr->getOffset().setVerticalIndirectionAccessID(mutatedFields_[expr->getName()]);
    }
  }
  std::map<std::string, int> mutatedFields_;

public:
  accessMutator(std::map<std::string, int> mutatedFields) : mutatedFields_(mutatedFields) {}
};

void injectRedirectedReads(std::shared_ptr<dawn::iir::StencilInstantiation> stencilInstantiation) {
  // for each field being read introduce another field to carry out the indirection
  for(auto& stencil : stencilInstantiation->getStencils()) {
    std::map<std::string, int> mutatedFields;
    for(auto& field : stencil->getOrderedFields()) {
      if(field.second.field.getReadExtents().has_value()) {
        int accessID = stencilInstantiation->getMetaData().addField(
            dawn::iir::FieldAccessType::APIField, field.second.Name + "_indirection",
            dawn::sir::FieldDimensions(field.second.field.getFieldDimensions()), std::nullopt);
        mutatedFields.insert({field.second.Name, accessID});
      }
    }

    // make each read access an indirected read access using the prepared fields
    accessMutator mutator(mutatedFields);
    stencil->accept(mutator);

    // this means the accesses of the statements changed. recompute them.
    std::vector<std::shared_ptr<dawn::ast::Stmt>> stmtsVec =
        dawn::iterateIIROverStmt(*stencilInstantiation->getIIR());
    dawn::ArrayRef<std::shared_ptr<dawn::ast::Stmt>> stmts(stmtsVec.data(), stmtsVec.size());
    dawn::computeAccesses(stencilInstantiation->getMetaData(), stmts);

    // this info needs to be propagated updwards
    for(const auto& doMethodPtr :
        dawn::iterateIIROver<dawn::iir::DoMethod>(*(stencilInstantiation->getIIR()))) {
      doMethodPtr->update(dawn::iir::NodeUpdateType::levelAndTreeAbove);
    }
  }
}