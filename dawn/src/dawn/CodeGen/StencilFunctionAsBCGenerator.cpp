#include "dawn/CodeGen/StencilFunctionAsBCGenerator.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/StencilInstantiation.h"

namespace dawn {
namespace codegen {

std::string
StencilFunctionAsBCGenerator::getName(const std::shared_ptr<iir::VarDeclStmt>& stmt) const {
  return metadata_.getFieldNameFromAccessID(iir::getAccessID(stmt));
}

std::string StencilFunctionAsBCGenerator::getName(const std::shared_ptr<iir::Expr>& expr) const {
  return metadata_.getFieldNameFromAccessID(iir::getAccessID(expr));
}

void StencilFunctionAsBCGenerator::visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) {
  expr->getName();
  auto getArgumentIndex = [&](const std::string& name) {
    size_t pos =
        std::distance(function_->Args.begin(),
                      std::find_if(function_->Args.begin(), function_->Args.end(),
                                   [&](const std::shared_ptr<sir::StencilFunctionArg>& arg) {
                                     return arg->Name == name;
                                   }));

    DAWN_ASSERT_MSG(pos < function_->Args.size(), "");
    return pos;
  };
  ss_ << dawn::format(
      "data_field_%i(%s)", getArgumentIndex(expr->getName()),
      toString(ast::cartesian, expr->getOffset(), ", ", [&](std::string const& name, int offset) {
        return name + "+" + std::to_string(offset);
      }));
}

void StencilFunctionAsBCGenerator::visit(const std::shared_ptr<iir::VarAccessExpr>& expr) {
  if(metadata_.isAccessType(iir::FieldAccessType::FAT_GlobalVariable, iir::getAccessID(expr)))
    ss_ << "m_globals.";

  ss_ << getName(expr);

  if(expr->isArrayAccess()) {
    ss_ << "[";
    expr->getIndex()->accept(*this);
    ss_ << "]";
  }
}

void BCGenerator::generate(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt) {
  const auto& hExtents = iir::extent_cast<iir::CartesianExtent const&>(
      metadata_.getBoundaryConditionExtentsFromBCStmt(stmt).horizontalExtent());
  const auto& vExtents = metadata_.getBoundaryConditionExtentsFromBCStmt(stmt).verticalExtent();
  int haloIMinus = std::abs(hExtents.iMinus());
  int haloIPlus = std::abs(hExtents.iPlus());
  int haloJMinus = std::abs(hExtents.jMinus());
  int haloJPlus = std::abs(hExtents.jPlus());
  int haloKMinus = std::abs(vExtents.minus());
  int haloKPlus = std::abs(vExtents.plus());
  std::string fieldname = stmt->getFields()[0];

  // Set up the halos
  std::string halosetup = dawn::format(
      "gridtools::array< gridtools::halo_descriptor, 3 > halos;\n"
      "halos[0] =gridtools::halo_descriptor(%i, %i, "
      "%s.get_storage_info_ptr()->template begin<0>(),%s.get_storage_info_ptr()->template "
      "end<0>(), "
      "%s.get_storage_info_ptr()->template total_length<0>());\nhalos[1] = "
      "gridtools::halo_descriptor(%i, "
      "%i, "
      "%s.get_storage_info_ptr()->template begin<1>(),%s.get_storage_info_ptr()->template "
      "end<1>(), "
      "%s.get_storage_info_ptr()->template total_length<1>());\nhalos[2] = "
      "gridtools::halo_descriptor(%i, "
      "%i, "
      "%s.get_storage_info_ptr()->template begin<2>(),%s.get_storage_info_ptr()->template "
      "end<2>(), "
      "%s.get_storage_info_ptr()->template total_length<2>());\n",
      haloIMinus, haloIPlus, fieldname, fieldname, fieldname, haloJMinus, haloJPlus, fieldname,
      fieldname, fieldname, haloKMinus, haloKPlus, fieldname, fieldname, fieldname);
  std::string makeView = "";

  // Create the views for the fields
  for(int i = 0; i < stmt->getFields().size(); ++i) {
    auto fieldName = stmt->getFields()[i];
    makeView +=
        dawn::format("auto %s_view = GT_BACKEND_DECISION_viewmaker(%s);\n", fieldName, fieldName);
  }
  std::string bcapply = "GT_BACKEND_DECISION_bcapply<" + stmt->getFunctor() + " >(halos, " +
                        stmt->getFunctor() + "()).apply(";
  for(int i = 0; i < stmt->getFields().size(); ++i) {
    bcapply += stmt->getFields()[i] + "_view";
    if(i < stmt->getFields().size() - 1) {
      bcapply += ", ";
    }
  }
  bcapply += ");\n";

  ss_ << halosetup;
  ss_ << makeView;
  ss_ << bcapply;
}
} // namespace codegen
} // namespace dawn
