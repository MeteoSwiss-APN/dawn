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
#include "dawn/SIR/SIR.h"
#include "dawn/SIR/SIRSerializer.h"
#include "test/unit-test/dawn/Optimizer/Passes/TestEnvironment.h"
#include <fstream>
#include <gtest/gtest.h>
#include <streambuf>

using namespace dawn;

namespace {

TEST(ComputeStageExtents, hori_diff_stencil_01) {
  std::string filename = TestEnvironment::path_ + "/hori_diff_stencil_01.sir";
  std::ifstream file(filename);
  DAWN_ASSERT_MSG((file.good()), std::string("File " + filename + " does not exists").c_str());

  std::string jsonstr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  std::shared_ptr<SIR> sir = SIRSerializer::deserializeFromString(jsonstr, SIRSerializer::SK_Json);

  // Prepare options
  std::unique_ptr<dawn::Options> compileOptions;

  // Run the compiler
  dawn::DawnCompiler compiler(compileOptions.get());
  std::unique_ptr<OptimizerContext> optimizer = compiler.runOptimizer(sir.get());
  // Report diganostics
  if(compiler.getDiagnostics().hasDiags()) {
    for(const auto& diag : compiler.getDiagnostics().getQueue())
      std::cerr << "Compilation Error " << diag->getMessage() << std::endl;
    throw std::runtime_error("compilation failed");
  }

  DAWN_ASSERT_MSG((optimizer->getStencilInstantiationMap().count("hori_diff_stencil")),
                  "hori_diff_stencil not found in sir");

  auto stencils = optimizer->getStencilInstantiationMap()["hori_diff_stencil"]->getStencils();
  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 2));
  ASSERT_TRUE((stencil->getStage(0)->getExtents() == Extents{-1, 1, -1, 1, 0, 0}));
  ASSERT_TRUE((stencil->getStage(1)->getExtents() == Extents{0, 0, 0, 0, 0, 0}));
}

TEST(ComputeStageExtents, hori_diff_stencil_02) {
  std::string filename = TestEnvironment::path_ + "/hori_diff_stencil_02.sir";
  std::ifstream file(filename);
  DAWN_ASSERT_MSG((file.good()), std::string("File " + filename + " does not exists").c_str());

  std::string jsonstr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  std::shared_ptr<SIR> sir = SIRSerializer::deserializeFromString(jsonstr, SIRSerializer::SK_Json);

  // Prepare options
  std::unique_ptr<dawn::Options> compileOptions;

  // Run the compiler
  dawn::DawnCompiler compiler(compileOptions.get());
  std::unique_ptr<OptimizerContext> optimizer = compiler.runOptimizer(sir.get());
  // Report diganostics
  if(compiler.getDiagnostics().hasDiags()) {
    for(const auto& diag : compiler.getDiagnostics().getQueue())
      std::cerr << "Compilation Error " << diag->getMessage() << std::endl;
    throw std::runtime_error("compilation failed");
  }

  DAWN_ASSERT_MSG((optimizer->getStencilInstantiationMap().count("hori_diff_stencil")),
                  "hori_diff_stencil not found in sir");

  auto stencils = optimizer->getStencilInstantiationMap()["hori_diff_stencil"]->getStencils();
  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 3));
  ASSERT_TRUE((stencil->getStage(0)->getExtents() == Extents{-1, 1, -1, 1, 0, 0}));
  ASSERT_TRUE((stencil->getStage(1)->getExtents() == Extents{-1, 0, -1, 0, 0, 0}));
  ASSERT_TRUE((stencil->getStage(2)->getExtents() == Extents{0, 0, 0, 0, 0, 0}));
}
TEST(ComputeStageExtents, hori_diff_stencil_03) {
  std::string filename = TestEnvironment::path_ + "/hori_diff_stencil_03.sir";
  std::ifstream file(filename);
  DAWN_ASSERT_MSG((file.good()), std::string("File " + filename + " does not exists").c_str());

  std::string jsonstr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  std::shared_ptr<SIR> sir = SIRSerializer::deserializeFromString(jsonstr, SIRSerializer::SK_Json);

  // Prepare options
  std::unique_ptr<dawn::Options> compileOptions;

  // Run the compiler
  dawn::DawnCompiler compiler(compileOptions.get());
  std::unique_ptr<OptimizerContext> optimizer = compiler.runOptimizer(sir.get());
  // Report diganostics
  if(compiler.getDiagnostics().hasDiags()) {
    for(const auto& diag : compiler.getDiagnostics().getQueue())
      std::cerr << "Compilation Error " << diag->getMessage() << std::endl;
    throw std::runtime_error("compilation failed");
  }

  DAWN_ASSERT_MSG((optimizer->getStencilInstantiationMap().count("hori_diff_stencil")),
                  "hori_diff_stencil not found in sir");

  auto stencils = optimizer->getStencilInstantiationMap()["hori_diff_stencil"]->getStencils();
  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 4));
  ASSERT_TRUE((stencil->getStage(0)->getExtents() == Extents{-1, 1, -1, 2, 0, 0}));
  ASSERT_TRUE((stencil->getStage(1)->getExtents() == Extents{-1, 0, -1, 1, 0, 0}));
  ASSERT_TRUE((stencil->getStage(2)->getExtents() == Extents{0, 0, 0, 1, 0, 0}));
  ASSERT_TRUE((stencil->getStage(3)->getExtents() == Extents{0, 0, 0, 0, 0, 0}));
}

TEST(ComputeStageExtents, hori_diff_stencil_04) {
  std::string filename = TestEnvironment::path_ + "/hori_diff_stencil_04.sir";
  std::ifstream file(filename);
  DAWN_ASSERT_MSG((file.good()), std::string("File " + filename + " does not exists").c_str());

  std::string jsonstr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  std::shared_ptr<SIR> sir = SIRSerializer::deserializeFromString(jsonstr, SIRSerializer::SK_Json);

  // Prepare options
  std::unique_ptr<dawn::Options> compileOptions = make_unique<Options>();

  compileOptions->MaxHaloPoints = 4;

  // Run the compiler
  dawn::DawnCompiler compiler(compileOptions.get());
  std::unique_ptr<OptimizerContext> optimizer = compiler.runOptimizer(sir.get());
  // Report diganostics
  if(compiler.getDiagnostics().hasDiags()) {
    for(const auto& diag : compiler.getDiagnostics().getQueue())
      std::cerr << "Compilation Error " << diag->getMessage() << std::endl;
    throw std::runtime_error("compilation failed");
  }

  DAWN_ASSERT_MSG((optimizer->getStencilInstantiationMap().count("hori_diff_stencil")),
                  "hori_diff_stencil not found in sir");

  auto stencils = optimizer->getStencilInstantiationMap()["hori_diff_stencil"]->getStencils();
  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 4));
  ASSERT_TRUE((stencil->getStage(0)->getExtents() == Extents{-2, 3, -2, 1, 0, 0}));
  ASSERT_TRUE((stencil->getStage(1)->getExtents() == Extents{-1, 1, -1, 0, 0, 0}));
  ASSERT_TRUE((stencil->getStage(2)->getExtents() == Extents{0, 0, -1, 0, 0, 0}));
  ASSERT_TRUE((stencil->getStage(3)->getExtents() == Extents{0, 0, 0, 0, 0, 0}));
}

TEST(ComputeStageExtents, hori_diff_stencil_05) {
  std::string filename = TestEnvironment::path_ + "/hori_diff_stencil_05.sir";
  std::ifstream file(filename);
  DAWN_ASSERT_MSG((file.good()), std::string("File " + filename + " does not exists").c_str());

  std::string jsonstr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  std::shared_ptr<SIR> sir = SIRSerializer::deserializeFromString(jsonstr, SIRSerializer::SK_Json);

  // Prepare options
  std::unique_ptr<dawn::Options> compileOptions = make_unique<Options>();

  compileOptions->MaxHaloPoints = 4;

  // Run the compiler
  dawn::DawnCompiler compiler(compileOptions.get());
  std::unique_ptr<OptimizerContext> optimizer = compiler.runOptimizer(sir.get());
  // Report diganostics
  if(compiler.getDiagnostics().hasDiags()) {
    for(const auto& diag : compiler.getDiagnostics().getQueue())
      std::cerr << "Compilation Error " << diag->getMessage() << std::endl;
    throw std::runtime_error("compilation failed");
  }

  DAWN_ASSERT_MSG((optimizer->getStencilInstantiationMap().count("hori_diff_stencil")),
                  "hori_diff_stencil not found in sir");

  auto stencils = optimizer->getStencilInstantiationMap()["hori_diff_stencil"]->getStencils();
  ASSERT_TRUE((stencils.size() == 1));
  std::shared_ptr<Stencil> stencil = stencils[0];

  ASSERT_TRUE((stencil->getNumStages() == 4));
  ASSERT_TRUE((stencil->getStage(0)->getExtents() == Extents{-2, 3, -2, 1, 0, 0}));
  ASSERT_TRUE((stencil->getStage(1)->getExtents() == Extents{-1, 1, -1, 0, 0, 0}));
  ASSERT_TRUE((stencil->getStage(2)->getExtents() == Extents{0, 1, -1, 0, 0, 0}));
  ASSERT_TRUE((stencil->getStage(3)->getExtents() == Extents{0, 0, 0, 0, 0, 0}));
}

} // anonymous namespace
