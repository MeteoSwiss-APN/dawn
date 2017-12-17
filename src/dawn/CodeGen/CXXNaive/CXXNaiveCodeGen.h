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

#include "dawn/CodeGen/CodeGen.h"
#include "dawn/Optimizer/Interval.h"
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace dawn {
class StencilInstantiation;
class OptimizerContext;

namespace codegen {
namespace cxxnaive {

/// @brief GridTools C++ code generation for the gridtools_clang DSL
/// @ingroup codegen
class CXXNaiveCodeGen : public CodeGen {
public:
  CXXNaiveCodeGen(dawn::OptimizerContext* context);
  virtual ~CXXNaiveCodeGen();
  virtual std::unique_ptr<TranslationUnit> generateCode() override;

private:
  std::string generateStencilInstantiation(const dawn::StencilInstantiation* stencilInstantiation);
};
} // namespace cxxnaive
} // namespace codegen
} // namespace dawn
