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

class TestCodeGenCuda : public TestCodeGen {};

TEST_F(TestCodeGenCuda, GlobalIndexStencil) {
  runTest(this->getGlobalIndexStencil(), "global_indexing.cu");
}

TEST_F(TestCodeGenCuda, NonOverlappingInterval) {
  runTest(this->getNonOverlappingInterval(), "nonoverlapping_stencil.cu");
}

TEST_F(TestCodeGenCuda, LaplacianStencil) {
  runTest(this->getLaplacianStencil(), "laplacian_stencil.cu");
}

TEST_F(TestCodeGenCuda, ConditionalStencil) {
  runTest(this->getConditionalStencil(), "conditional_stencil.cu");
}

TEST_F(TestCodeGenCuda, DzCStencil) {
  runTest(this->getStencilFromIIR("update_dz_c"), "update_dz_c.cu");
}

} // anonymous namespace
