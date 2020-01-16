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

#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Compiler/Options.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/IntegrityChecker.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/CodeGen/CodeGen.h"
#include "dawn/CodeGen/CXXNaive/CXXNaiveCodeGen.h"
#include "dawn/Support/DiagnosticsEngine.h"
#include "dawn/Unittest/IIRBuilder.h"
#include "dawn/Unittest/UnittestLogger.h"

#include <gtest/gtest.h>

#include <cstring>
#include <fstream>

using namespace dawn;

using stencilInstantiationContext =
  std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>;

namespace {

void dump(std::ostream& os, dawn::codegen::stencilInstantiationContext& ctx) {
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

std::string read(const std::string& file) {
  std::ifstream is(file);
  std::string str((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
  return str;
}

std::shared_ptr<SIR> deserialize(const std::string& file) {
  std::string json = read(file);
  return SIRSerializer::deserializeFromString(json, SIRSerializer::Format::Json);
}

stencilInstantiationContext compile(std::shared_ptr<SIR> sir) {
  std::unique_ptr<dawn::Options> options;
  DawnCompiler compiler(options.get());
  auto optimizer = compiler.runOptimizer(sir);

  if(compiler.getDiagnostics().hasDiags()) {
    for (const auto &diag : compiler.getDiagnostics().getQueue()) {
      std::cerr << "Compilation Error " << diag->getMessage() << std::endl;
    }
    throw std::runtime_error("Compilation failed");
  }

  return optimizer->getStencilInstantiationMap();
}

TEST(CodeGenNaiveTest, GlobalsOptimizedAway) {
  std::shared_ptr<SIR> sir = deserialize("input/globals_opt_away.sir");
  auto stencil_inst = compile(sir);
  ASSERT_FALSE(stencil_inst.empty());

  IntegrityChecker checker(stencil_inst.begin()->second.get());
  try {
    checker.run();
    FAIL() << "Semantic error not thrown";
  } catch (SemanticError& error) {
    SUCCEED();
  }

  std::ostringstream oss;
  dump(oss, stencil_inst);
}

} // anonymous namespace
