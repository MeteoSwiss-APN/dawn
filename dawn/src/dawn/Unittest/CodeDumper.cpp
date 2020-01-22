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

#include "dawn/Unittest/CodeDumper.h"

namespace dawn {

void CodeDumper::dumpNaive(std::ostream& os, dawn::codegen::stencilInstantiationContext& ctx) {
  using CG = dawn::codegen::cxxnaive::CXXNaiveCodeGen;
  dawn::DiagnosticsEngine diagnostics;
  CG generator(ctx, diagnostics, 0);
  auto tu = generator.generateCode();

  std::ostringstream ss;
  for(auto const& macroDefine : tu->getPPDefines())
    ss << macroDefine << "\n";

  ss << tu->getGlobals();
  for(auto const& s : tu->getStencils())
    ss << s.second;
  os << ss.str();
}

void CodeDumper::dumpCuda(std::ostream& os, dawn::codegen::stencilInstantiationContext& ctx) {
  using CG = dawn::codegen::cuda::CudaCodeGen;
  dawn::DiagnosticsEngine diagnostics;
  CG generator(ctx, diagnostics, 0, 0, 0, {0, 0, 0});
  auto tu = generator.generateCode();

  std::ostringstream ss;
  for(auto const& macroDefine : tu->getPPDefines())
    ss << macroDefine << "\n";

  ss << tu->getGlobals();
  for(auto const& s : tu->getStencils())
    ss << s.second;
  os << ss.str();
}

} // namespace dawn
