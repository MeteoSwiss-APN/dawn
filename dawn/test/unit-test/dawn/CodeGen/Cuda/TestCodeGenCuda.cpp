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

#include "../TestCodeGen.h"

namespace {

constexpr auto backend = dawn::codegen::Backend::CUDA;

class TestCodeGenCuda : public TestCodeGen {};

TEST_F(TestCodeGenCuda, GlobalIndexStencil) {
  runTest(this->getGlobalIndexStencil(), backend, "global_indexing.cu");
}

TEST_F(TestCodeGenCuda, NonOverlappingInterval) {
  runTest(this->getNonOverlappingInterval(), backend, "nonoverlapping_stencil.cu");
}

TEST_F(TestCodeGenCuda, LaplacianStencil) {
  runTest(this->getLaplacianStencil(), backend, "laplacian_stencil.cu");
}

TEST_F(TestCodeGenCuda, ConditionalStencil) {
  runTest(this->getConditionalStencil(), backend, "conditional_stencil.cu");
}

} // anonymous namespace
