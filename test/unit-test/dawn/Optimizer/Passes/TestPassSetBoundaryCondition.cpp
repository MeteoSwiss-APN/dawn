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
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/SIR/SIRSerializer.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"
#include <fstream>
#include <gtest/gtest.h>
#include <streambuf>

using namespace dawn;

namespace {
class StencilSplitAnalyzer : public ::testing::Test {
  std::unique_ptr<dawn::Options> compileOptions_;

  dawn::DawnCompiler compiler_;

protected:
  StencilSplitAnalyzer() : compiler_(compileOptions_.get()) {}
  virtual void SetUp() {}

  std::shared_ptr<StencilInstantiation> loadTest(std::string sirFilename, bool splitStencils) {
    return loadTest(sirFilename, splitStencils, -1);
  }

  std::shared_ptr<StencilInstantiation> loadTest(std::string sirFilename, bool splitStencils,
                                                 int maxfields) {
    std::string filename = TestEnvironment::path_ + "/" + sirFilename;
    std::ifstream file(filename);
    DAWN_ASSERT_MSG((file.good()), std::string("File " + filename + " does not exists").c_str());

    std::string jsonstr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::shared_ptr<SIR> sir =
        SIRSerializer::deserializeFromString(jsonstr, SIRSerializer::SK_Json);

    // Set specific compiler options:
    if(splitStencils)
      compiler_.getOptions().SplitStencils = true;
    if(maxfields != -1)
      compiler_.getOptions().MaxFieldsPerStencil = maxfields;

    // Run the optimization
    std::unique_ptr<OptimizerContext> optimizer = compiler_.runOptimizer(sir);

    // Report diganostics
    if(compiler_.getDiagnostics().hasDiags()) {
      for(const auto& diag : compiler_.getDiagnostics().getQueue())
        std::cerr << "Compilation Error " << diag->getMessage() << std::endl;
      throw std::runtime_error("compilation failed");
    }

    DAWN_ASSERT_MSG((optimizer->getStencilInstantiationMap().count("SplitStencil")),
                    "SplitStencil not found in sir");
    return std::move(optimizer->getStencilInstantiationMap()["SplitStencil"]);
  }
};

class BCFinder : public ASTVisitorForwarding {
public:
  using Base = ASTVisitorForwarding;
  BCFinder() : BCsFound_(0) {}
  void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) {
    BCsFound_++;
    Base::visit(stmt);
  }
  void resetFinder() { BCsFound_ = 0; }

  int reportBCsFound() { return BCsFound_; }

private:
  int BCsFound_;
};

TEST_F(StencilSplitAnalyzer, test_no_bc_inserted) {
  std::shared_ptr<StencilInstantiation> test =
      loadTest("boundary_condition_test_stencil_01.sir", false);
  ASSERT_TRUE((test->getBoundaryConditions().size() == 1));
  BCFinder myvisitor;
  for(const auto& stmt : test->getStencilDescStatements()) {
    stmt->ASTStmt->accept(myvisitor);
  }
  ASSERT_TRUE((myvisitor.reportBCsFound() == 0));
}

// An unused BC has no extents to it
TEST_F(StencilSplitAnalyzer, test_unused_bc) {
  std::shared_ptr<StencilInstantiation> test =
      loadTest("boundary_condition_test_stencil_02.sir", false);
  ASSERT_TRUE(test->getBoundaryConditions().count("out"));
  auto bc = test->getBoundaryConditions().find("out")->second;
  ASSERT_TRUE((test->getBoundaryConditionToExtentsMap().count(bc) == 0));
}

TEST_F(StencilSplitAnalyzer, test_bc_extent_calc) {
  std::shared_ptr<StencilInstantiation> test =
      loadTest("boundary_condition_test_stencil_01.sir", true, 2);
  ASSERT_TRUE((test->getBoundaryConditions().size() == 1));
  BCFinder myvisitor;
  for(const auto& stmt : test->getStencilDescStatements()) {
    stmt->ASTStmt->accept(myvisitor);
  }
  ASSERT_TRUE((myvisitor.reportBCsFound() == 1));
  ASSERT_TRUE(test->getBoundaryConditions().count("intermediate"));
  auto bc = test->getBoundaryConditions().find("intermediate")->second;
  ASSERT_TRUE((test->getBoundaryConditionToExtentsMap()[bc] == Extents{-1, 1, 0, 0, 0, 0}));
}

TEST_F(StencilSplitAnalyzer, test_two_bc) {
  std::shared_ptr<StencilInstantiation> test =
      loadTest("boundary_condition_test_stencil_03.sir", true, 2);
  ASSERT_TRUE((test->getBoundaryConditions().size() == 2));
  ASSERT_TRUE(test->getBoundaryConditions().count("intermediate"));
  auto bcfoo = test->getBoundaryConditions().find("intermediate")->second;
  ASSERT_TRUE((test->getBoundaryConditionToExtentsMap()[bcfoo] == Extents{-1, 1, 0, 0, 0, 0}));
  ASSERT_TRUE(test->getBoundaryConditions().count("out"));
  auto bcbar = test->getBoundaryConditions().find("out")->second;
  ASSERT_TRUE((test->getBoundaryConditionToExtentsMap().count(bcbar) == 0));
}

} // anonymous namespace
