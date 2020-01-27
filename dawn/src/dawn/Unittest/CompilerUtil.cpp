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

#include "dawn/Unittest/CompilerUtil.h"

namespace dawn {

const std::shared_ptr<iir::StencilInstantiation>
CompilerUtil::load(const std::string& iirFilename,
                   const dawn::OptimizerContext::OptimizerContextOptions& options,
                   std::unique_ptr<OptimizerContext>& context) {
  std::ifstream file(iirFilename.c_str());
  DAWN_ASSERT_MSG((file.good()), std::string("File '" + iirFilename + "' does not exist").c_str());

  dawn::DiagnosticsEngine diag;
  std::shared_ptr<SIR> sir = std::make_shared<SIR>(ast::GridType::Cartesian);
  context = std::make_unique<OptimizerContext>(diag, options, sir);

  std::string jsonstr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  std::shared_ptr<iir::StencilInstantiation> stencilInstantion =
      IIRSerializer::deserializeFromString(jsonstr, context.get());

  return stencilInstantion;
}

stencilInstantiationContext CompilerUtil::compile(const std::shared_ptr<SIR>& sir) {
  dawn::Options options;
  DawnCompiler compiler(options);
  auto optimizer = compiler.runOptimizer(sir);

  if(compiler.getDiagnostics().hasDiags()) {
    for(const auto& diag : compiler.getDiagnostics().getQueue()) {
      std::cerr << "Compilation Error " << diag->getMessage() << std::endl;
    }
    throw std::runtime_error("Compilation failed");
  }

  return optimizer->getStencilInstantiationMap();
}

stencilInstantiationContext CompilerUtil::compile(const std::string& sirFile) {
  return compile(SIRSerializer::deserialize("input/globals_opt_away.sir"));
}

void CompilerUtil::dumpNaive(std::ostream& os, dawn::codegen::stencilInstantiationContext& ctx) {
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

void CompilerUtil::dumpCuda(std::ostream& os, dawn::codegen::stencilInstantiationContext& ctx) {
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
