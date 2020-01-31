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
#include "dawn/CodeGen/CXXNaive/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/CodeGen.h"
#include "dawn/CodeGen/Cuda/CudaCodeGen.h"

namespace dawn {

dawn::DiagnosticsEngine CompilerUtil::diag_;

void CompilerUtil::load(const std::string& sirFilename, std::shared_ptr<SIR>& sir) {
  sir = SIRSerializer::deserialize(sirFilename);
}

void CompilerUtil::load(const std::string& iirFilename,
                        const dawn::OptimizerContext::OptimizerContextOptions& options,
                        std::unique_ptr<OptimizerContext>& context,
                        std::shared_ptr<iir::StencilInstantiation>& instantiation,
                        const std::string& envPath) {
  std::string filename = envPath;
  if(!filename.empty())
    filename += "/";
  filename += iirFilename;

  std::shared_ptr<SIR> sir = std::make_shared<SIR>(ast::GridType::Cartesian);
  context = std::make_unique<OptimizerContext>(diag_, options, sir);
  instantiation = IIRSerializer::deserialize(iirFilename);
}

void CompilerUtil::lower(const std::shared_ptr<dawn::SIR>& sir,
                         std::unique_ptr<OptimizerContext>& context,
                         std::shared_ptr<iir::StencilInstantiation>& instantiation) {
  dawn::OptimizerContext::OptimizerContextOptions options;
  context = std::make_unique<OptimizerContext>(diag_, options, sir);
  std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>& map =
      context->getStencilInstantiationMap();
  instantiation = map.begin()->second;
}

void CompilerUtil::lower(const std::string& sirFilename,
                         std::unique_ptr<OptimizerContext>& context,
                         std::shared_ptr<iir::StencilInstantiation>& instantiation,
                         const std::string& envPath) {
  std::string filename = envPath;
  if(!filename.empty())
    filename += "/";
  filename += sirFilename;

  std::shared_ptr<dawn::SIR> sir;
  load(filename, sir);
  lower(sir, context, instantiation);
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
  std::shared_ptr<SIR> sir;
  load(sirFile, sir);
  return compile(sir);
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
