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

#ifndef DAWN_CODEGEN_CODEGENPROPERTIES_H
#define DAWN_CODEGEN_CODEGENPROPERTIES_H

#include "dawn/Support/Assert.h"
#include <memory>
#include <unordered_map>

namespace dawn {
namespace codegen {
// @brief context of a stencil body
// (pure stencil or a stencil function)
enum class StencilContext { SC_Stencil = 0, SC_StencilFunction };

struct StencilProperties {
  std::unordered_map<std::string, std::string> paramNameToType_;
  const int id_;
  const std::string name_;
  StencilProperties(const int id, const std::string name) : id_(id), name_(name) {}
};

struct CodeGenProperties {
  struct Impl {
    std::unordered_map<std::string, std::shared_ptr<StencilProperties>> stencilProps_;
    std::unordered_map<size_t, std::string> stencilIDToName_;
  };

  std::unordered_map<std::string, std::string> paramNameToType_;
  std::unordered_map<size_t, std::string> paramPositionIdxToName_;

  std::array<Impl, 2> stencilContextProperties_;

  void insertParam(const size_t paramPosition, std::string paramName, std::string paramType);

  std::string getParamType(const std::string paramName) const;

  std::unordered_map<std::string, std::shared_ptr<StencilProperties>>&
  stencilProperties(StencilContext context);

  std::shared_ptr<StencilProperties> insertStencil(StencilContext context, const int id,
                                                   const std::string name);

  std::shared_ptr<StencilProperties> getStencilProperties(StencilContext context,
                                                          const std::string name);

  std::shared_ptr<StencilProperties> getStencilProperties(StencilContext context, const int id);

  std::string getStencilName(StencilContext context, const size_t id) const;

private:
  std::shared_ptr<StencilProperties> insertStencil(Impl& impl, const size_t id,
                                                   const std::string name);
};
} // namespace codegen
} // namespace dawn
#endif
