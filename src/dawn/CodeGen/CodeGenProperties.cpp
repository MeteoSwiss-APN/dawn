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

#include "dawn/CodeGen/CodeGenProperties.h"
#include <memory>

namespace dawn {
namespace codegen {

void CodeGenProperties::insertParam(const size_t paramPosition, std::string paramName,
                                    std::string paramType) {
  DAWN_ASSERT_MSG(!paramPositionIdxToName_.count(paramPosition), "parameter already inserted");
  paramPositionIdxToName_[paramPosition] = paramName;
  DAWN_ASSERT_MSG(!paramNameToType_.count(paramName), "parameter already inserted");
  paramNameToType_[paramName] = paramType;
}

std::string CodeGenProperties::getParamType(const std::string paramName) const {
  DAWN_ASSERT_MSG(paramNameToType_.count(paramName),
                  std::string("parameter " + paramName + " not found").c_str());
  return paramNameToType_.at(paramName);
}

std::unordered_map<std::string, std::shared_ptr<StencilProperties>>&
CodeGenProperties::stencilProperties(StencilContext context) {
  return stencilContextProperties_[static_cast<int>(context)].stencilProps_;
}

std::shared_ptr<StencilProperties>
CodeGenProperties::insertStencil(StencilContext context, const int id, const std::string name) {
  return insertStencil(stencilContextProperties_[static_cast<int>(context)], id, name);
}

std::shared_ptr<StencilProperties> CodeGenProperties::getStencilProperties(StencilContext context,
                                                                           const std::string name) {
  DAWN_ASSERT_MSG(stencilContextProperties_[static_cast<int>(context)].stencilProps_.count(name),
                  "stencil name not found");
  return stencilContextProperties_[static_cast<int>(context)].stencilProps_[name];
}

std::shared_ptr<StencilProperties> CodeGenProperties::getStencilProperties(StencilContext context,
                                                                           const int id) {
  return getStencilProperties(context, getStencilName(context, id));
}

std::string CodeGenProperties::getStencilName(StencilContext context, const size_t id) const {
  DAWN_ASSERT_MSG(stencilContextProperties_[static_cast<int>(context)].stencilIDToName_.count(id),
                  "id of stencil not found");
  return stencilContextProperties_[static_cast<int>(context)].stencilIDToName_.at(id);
}

std::shared_ptr<StencilProperties> CodeGenProperties::insertStencil(Impl& impl, const size_t id,
                                                                    const std::string name) {
  DAWN_ASSERT_MSG(!impl.stencilIDToName_.count(id), "stencil already inserted");
  impl.stencilIDToName_[id] = name;
  DAWN_ASSERT_MSG(!impl.stencilProps_.count(name), "stencil already inserted");
  impl.stencilProps_[name] = std::make_shared<StencilProperties>(id, name);
  return impl.stencilProps_[name];
}

} // namespace codegen
} // namespace dawn
