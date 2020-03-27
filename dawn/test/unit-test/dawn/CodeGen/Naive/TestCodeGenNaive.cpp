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

class TestCodeGenNaive : public TestCodeGen {};

TEST_F(TestCodeGenNaive, GlobalIndexStencil) {
  runTest(this->getGlobalIndexStencil(), "global_indexing.cpp");
}

TEST_F(TestCodeGenNaive, NonOverlappingInterval) {
  runTest(this->getNonOverlappingInterval(), "nonoverlapping_stencil.cpp");
}

TEST_F(TestCodeGenNaive, LaplacianStencil) {
  runTest(this->getLaplacianStencil(), "laplacian_stencil.cpp");
}

TEST_F(TestCodeGenNaive, ConditionalStencil) {
  runTest(this->getConditionalStencil(), "conditional_stencil.cpp");
}

TEST_F(TestCodeGenNaive, DzCStencil) {
  runTest(this->getStencilFromIIR("update_dz_c"), "update_dz_c.cpp");
}

} // anonymous namespace
