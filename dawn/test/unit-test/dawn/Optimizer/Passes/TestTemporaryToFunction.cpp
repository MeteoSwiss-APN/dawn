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

#include "dawn/CodeGen/CXXNaive/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/GridTools/GTCodeGen.h"
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "test/unit-test/dawn/Optimizer/TestEnvironment.h"
#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

class TemporaryToFunction : public ::testing::Test {

  dawn::DawnCompiler compiler_;

protected:
  TemporaryToFunction() {
    compiler_.getOptions().PassTmpToFunction = true;
    //    compiler_.getOptions().ReportPassTmpToFunction = true;
  }
  virtual void SetUp() {}

  std::shared_ptr<iir::StencilInstantiation> loadTest(std::string sirFilename) {

    std::string filename = TestEnvironment::path_ + "/" + sirFilename;
    std::ifstream file(filename);
    DAWN_ASSERT_MSG((file.good()), std::string("File " + filename + " does not exists").c_str());

    std::string jsonstr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::shared_ptr<SIR> sir =
        SIRSerializer::deserializeFromString(jsonstr, SIRSerializer::Format::Json);

    auto stencilInstantiationMap = compiler_.optimize(compiler_.parallelize(sir));
    // Report diganostics
    if(compiler_.getDiagnostics().hasDiags()) {
      for(const auto& diag : compiler_.getDiagnostics().getQueue())
        std::cerr << "Compilation Error " << diag->getMessage() << std::endl;
      throw std::runtime_error("compilation failed");
    }

    DAWN_ASSERT_MSG(stencilInstantiationMap.count("compute_extent_test_stencil"),
                    "compute_extent_test_stencil not found in sir");

    // // Generate code
    auto translationUnit = compiler_.generate(stencilInstantiationMap);
    DAWN_ASSERT(translationUnit);

    return stencilInstantiationMap["compute_extent_test_stencil"];
  }
};

} // anonymous namespace
