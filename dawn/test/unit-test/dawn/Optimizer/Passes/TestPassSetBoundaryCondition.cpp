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
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"
#include <fstream>
#include <gtest/gtest.h>
#include <streambuf>

using namespace dawn;

namespace {
class StencilSplitAnalyzer : public ::testing::Test {
  dawn::DawnCompiler compiler_;

protected:
  virtual void SetUp() {}

  std::shared_ptr<iir::StencilInstantiation> loadTest(std::string sirFilename, bool splitStencils) {
    return loadTest(sirFilename, splitStencils, -1);
  }

  std::shared_ptr<iir::StencilInstantiation> loadTest(std::string sirFilename, bool splitStencils,
                                                      int maxfields) {
    std::string filename = TestEnvironment::path_ + "/" + sirFilename;
    std::ifstream file(filename);
    DAWN_ASSERT_MSG((file.good()), std::string("File " + filename + " does not exists").c_str());

    std::string jsonstr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::shared_ptr<SIR> sir =
        SIRSerializer::deserializeFromString(jsonstr, SIRSerializer::Format::Json);

    // Set specific compiler options:
    if(splitStencils)
      compiler_.getOptions().SplitStencils = true;
    if(maxfields != -1)
      compiler_.getOptions().MaxFieldsPerStencil = maxfields;

    // Run the optimization
    std::unique_ptr<OptimizerContext> optimizer = compiler_.runOptimizer(sir);

    // Report diagnostics
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

class BCFinder : public iir::ASTVisitorForwarding {
public:
  using Base = iir::ASTVisitorForwarding;
  BCFinder() : BCsFound_(0) {}
  void visit(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt) {
    BCsFound_++;
    Base::visit(stmt);
  }
  void resetFinder() { BCsFound_ = 0; }

  int reportBCsFound() { return BCsFound_; }

private:
  int BCsFound_;
};

TEST_F(StencilSplitAnalyzer, DISABLED_test_no_bc_inserted) {
  std::shared_ptr<iir::StencilInstantiation> test =
      loadTest("boundary_condition_test_stencil_01.sir", false);
  ASSERT_TRUE((test->getMetaData().getFieldNameToBCMap().size() == 1));
  BCFinder myvisitor;
  for(const auto& stmt : test->getIIR()->getControlFlowDescriptor().getStatements()) {
    stmt->accept(myvisitor);
  }
  ASSERT_TRUE((myvisitor.reportBCsFound() == 0));
}

// The boundary condition tests disabled until boundary conditions are fixed and
// added to the c++-naive codegen backend.

// An unused BC has no extents to it
TEST_F(StencilSplitAnalyzer, DISABLED_test_unused_bc) {
  std::shared_ptr<iir::StencilInstantiation> test =
      loadTest("boundary_condition_test_stencil_02.sir", false);
  ASSERT_TRUE(test->getMetaData().getFieldNameToBCMap().count("out"));
  auto bc = test->getMetaData().getFieldNameToBCMap().find("out")->second;
  ASSERT_TRUE((!test->getMetaData().hasBoundaryConditionStmtToExtent(bc)));
}

TEST_F(StencilSplitAnalyzer, DISABLED_test_bc_extent_calc) {
  std::shared_ptr<iir::StencilInstantiation> test =
      loadTest("boundary_condition_test_stencil_01.sir", true, 2);
  ASSERT_TRUE((test->getMetaData().getFieldNameToBCMap().size() == 1));
  BCFinder myvisitor;
  for(const auto& stmt : test->getIIR()->getControlFlowDescriptor().getStatements()) {
    stmt->accept(myvisitor);
  }
  ASSERT_TRUE((myvisitor.reportBCsFound() == 1));
  ASSERT_TRUE(test->getMetaData().getFieldNameToBCMap().count("intermediate"));
  auto bc = test->getMetaData().getFieldNameToBCMap().find("intermediate")->second;
  ASSERT_TRUE((test->getMetaData().getBoundaryConditionExtentsFromBCStmt(bc) ==
               iir::Extents(ast::cartesian, -1, 1, 0, 0, 0, 0)));
} // namespace

TEST_F(StencilSplitAnalyzer, DISABLED_test_two_bc) {
  std::shared_ptr<iir::StencilInstantiation> test =
      loadTest("boundary_condition_test_stencil_03.sir", true, 2);
  ASSERT_TRUE((test->getMetaData().getFieldNameToBCMap().size() == 2));
  ASSERT_TRUE(test->getMetaData().getFieldNameToBCMap().count("intermediate"));
  auto bcfoo = test->getMetaData().getFieldNameToBCMap().find("intermediate")->second;
  ASSERT_TRUE((test->getMetaData().getBoundaryConditionExtentsFromBCStmt(bcfoo) ==
               iir::Extents(ast::cartesian, -1, 1, 0, 0, 0, 0)));
  ASSERT_TRUE(test->getMetaData().getFieldNameToBCMap().count("out"));
  auto bcbar = test->getMetaData().getFieldNameToBCMap().find("out")->second;
  ASSERT_TRUE((!test->getMetaData().hasBoundaryConditionStmtToExtent(bcbar)));
}

} // anonymous namespace
