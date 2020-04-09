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
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/PassMultiStageMerger.h"
#include "dawn/Optimizer/PassSetDependencyGraph.h"
#include "dawn/Optimizer/PassSetStageGraph.h"
#include "dawn/Optimizer/PassStageMerger.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"
#include <fstream>
#include <gtest/gtest.h>
#include <streambuf>
#include <string>

namespace dawn {

class TestFromSIR : public ::testing::Test {
  dawn::OptimizerContext::OptimizerContextOptions options_;
  DiagnosticsEngine diagnostics_;
  dawn::DawnCompiler compiler_;
  std::unique_ptr<OptimizerContext> context_;

protected:
  TestFromSIR() {
    context_ = std::make_unique<dawn::OptimizerContext>(diagnostics_, options_, nullptr);
  }

  std::shared_ptr<iir::StencilInstantiation> loadTest(std::string sirFilename) {
    std::string filename = sirFilename;
    if(!TestEnvironment::path_.empty())
      filename = TestEnvironment::path_ + "/" + filename;
    std::ifstream file(filename);
    DAWN_ASSERT_MSG((file.good()), std::string("File " + filename + " does not exists").c_str());

    std::string jsonstr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::shared_ptr<SIR> sir =
        SIRSerializer::deserializeFromString(jsonstr, SIRSerializer::Format::Json);
    auto stencilInstantiationMap = compiler_.lowerToIIR(sir);

    DAWN_ASSERT_MSG(stencilInstantiationMap.size() == 1, "unexpected number of stencils");
    DAWN_ASSERT_MSG(stencilInstantiationMap.count("compute_extent_test_stencil"),
                    "compute_extent_test_stencil not found in sir");
    auto instantiation = stencilInstantiationMap["compute_extent_test_stencil"];

    // Run stage graph pass
    PassSetStageGraph stageGraphPass(*context_);
    EXPECT_TRUE(stageGraphPass.run(instantiation));

    // Run dependency graph pass
    PassSetDependencyGraph dependencyGraphPass(*context_);
    EXPECT_TRUE(dependencyGraphPass.run(instantiation));

    // Run multistage merger pass
    PassMultiStageMerger multiStageMergerPass(*context_);
    EXPECT_TRUE(multiStageMergerPass.run(instantiation));

    // Run stage merger pass
    PassStageMerger stageMergerPass(*context_);
    EXPECT_TRUE(stageMergerPass.run(instantiation));

    // Recompute derived info...
    instantiation->computeDerivedInfo();
    IIRSerializer::serialize(filename + ".iir", instantiation);

    return instantiation;
  }
};

} // namespace dawn
