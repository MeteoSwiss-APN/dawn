#include "dawn/CodeGen/StencilFunctionAsBCGenerator.h"

namespace dawn {
namespace codegen {

void StencilFunctionAsBCGenerator::visit(const std::shared_ptr<FieldAccessExpr>& expr) {
  auto printOffset = [](const Array3i& argumentoffsets) {
    std::string retval = "";
    std::array<std::string, 3> dims{"i", "j", "k"};
    for(int i = 0; i < 3; ++i) {
      retval +=
          dims[i] + (argumentoffsets[i] != 0 ? " + " + std::to_string(argumentoffsets[i]) + ", "
                                             : (i < 2 ? ", " : ""));
    }
    return retval;
  };
  expr->getName();
  auto getArgumentIndex = [&](const std::string& name) {
    size_t pos =
        std::distance(function->Args.begin(),
                      std::find_if(function->Args.begin(), function->Args.end(),
                                   [&](const std::shared_ptr<sir::StencilFunctionArg>& arg) {
                                     return arg->Name == name;
                                   }));

    DAWN_ASSERT_MSG(pos < function->Args.size(), "");
    return pos;
  };
  ss_ << dawn::format("data_field_%i(%s)", getArgumentIndex(expr->getName()),
                      printOffset(expr->getOffset()));
}

void StencilFunctionAsBCGenerator::visit(const std::shared_ptr<VarAccessExpr>& expr) {
  if(instantiation_->isGlobalVariable(instantiation_->getAccessIDFromExpr(expr)))
    ss_ << "m_globals.";

  ss_ << getName(expr);

  if(expr->isArrayAccess()) {
    ss_ << "[";
    expr->getIndex()->accept(*this);
    ss_ << "]";
  }
}
}
}
