#include "dawn/CodeGen/StencilFunctionAsBCGenerator.h"
#include "dawn/IIR/StencilInstantiation.h"

namespace dawn {
namespace codegen {

std::string StencilFunctionAsBCGenerator::getName(const std::shared_ptr<iir::Stmt>& stmt) const {
  return metadata_.getFieldNameFromAccessID(metadata_.getAccessIDFromStmt(stmt));
}

std::string StencilFunctionAsBCGenerator::getName(const std::shared_ptr<iir::Expr>& expr) const {
  return metadata_.getFieldNameFromAccessID(metadata_.getAccessIDFromExpr(expr));
}

void StencilFunctionAsBCGenerator::visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) {
  auto printOffset = [](const ast::Offset& argOffset) {
    auto const& hoffset =
        ast::offset_cast<ast::StructuredOffset const&>(argOffset.horizontalOffset());
    auto const& voffset = argOffset.verticalOffset();

    std::string delim;

    std::ostringstream os;
    if(hoffset.offsetI()) {
      os << delim << " + " << hoffset.offsetI();
      delim = ", ";
    }
    if(hoffset.offsetJ()) {
      os << delim << " + " << hoffset.offsetJ();
      delim = ", ";
    }
    if(voffset) {
      os << delim << " + " << voffset;
    }
    return os.str();
  };
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
  ss_ << dawn::format("data_field_%i(%s)", getArgumentIndex(expr->getName()),
                      printOffset(expr->getOffset()));
}

void StencilFunctionAsBCGenerator::visit(const std::shared_ptr<iir::VarAccessExpr>& expr) {
  if(metadata_.isAccessType(iir::FieldAccessType::FAT_GlobalVariable,
                            metadata_.getAccessIDFromExpr(expr)))
    ss_ << "m_globals.";

  ss_ << getName(expr);

  if(expr->isArrayAccess()) {
    ss_ << "[";
    expr->getIndex()->accept(*this);
    ss_ << "]";
  }
}

void BCGenerator::generate(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt) {
  iir::Extents extents = metadata_.getBoundaryConditionExtentsFromBCStmt(stmt);
  int haloIMinus = abs(extents[0].Minus);
  int haloIPlus = abs(extents[0].Plus);
  int haloJMinus = abs(extents[1].Minus);
  int haloJPlus = abs(extents[1].Plus);
  int haloKMinus = abs(extents[2].Minus);
  int haloKPlus = abs(extents[2].Plus);
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
