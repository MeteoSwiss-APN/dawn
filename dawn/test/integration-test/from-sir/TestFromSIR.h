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
#include "dawn/Compiler/Driver.h"
#include "dawn/Compiler/Options.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/PassMultiStageMerger.h"
#include "dawn/Optimizer/PassSetDependencyGraph.h"
#include "dawn/Optimizer/PassSetStageGraph.h"
#include "dawn/Optimizer/PassSetSyncStage.h"
#include "dawn/Optimizer/PassStageMerger.h"
#include "dawn/Optimizer/PassTemporaryType.h"
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

  std::shared_ptr<iir::StencilInstantiation>
  loadTest(const std::string& sirFilename,
           const std::string& stencilName = "compute_extent_test_stencil") {
    std::string filename = sirFilename;
    if(!TestEnvironment::path_.empty())
      filename = TestEnvironment::path_ + "/" + filename;
    std::ifstream file(filename);
    DAWN_ASSERT_MSG((file.good()), std::string("File " + filename + " does not exists").c_str());

    std::string jsonstr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::shared_ptr<SIR> sir =
        SIRSerializer::deserializeFromString(jsonstr, SIRSerializer::Format::Json);
    auto stencilInstantiationMap = compiler_.lowerToIIR(sir);
    std::list<PassGroup> groups = defaultPassGroups();
    stencilInstantiationMap = compiler_.optimize(stencilInstantiationMap, groups);

    DAWN_ASSERT_MSG(stencilInstantiationMap.size() == 1, "unexpected number of stencils");
    DAWN_ASSERT_MSG(stencilInstantiationMap.count(stencilName),
                    (stencilName + " not found in sir").c_str());
    auto instantiation = stencilInstantiationMap[stencilName];

    //    // Recompute derived info...
    //    instantiation->computeDerivedInfo();
    //    IIRSerializer::serialize(filename + ".iir", instantiation);

    return instantiation;
  }
};

} // namespace dawn
