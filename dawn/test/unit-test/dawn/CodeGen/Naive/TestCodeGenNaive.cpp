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

namespace dawn {
namespace iir {

constexpr auto backend = dawn::codegen::Backend::CXXNaive;

class TestCodeGenNaive : public TestCodeGen {};

TEST_F(TestCodeGenNaive, GlobalIndexStencil) {
  runTest(this->getGlobalIndexStencil(), backend, "global_indexing.cpp");
}

TEST_F(TestCodeGenNaive, NonOverlappingInterval) {
  runTest(this->getNonOverlappingInterval(), backend, "nonoverlapping_stencil.cpp");
}

TEST_F(TestCodeGenNaive, LaplacianStencil) {
  runTest(this->getLaplacianStencil(), backend, "laplacian_stencil.cpp");
}

TEST_F(TestCodeGenNaive, ConditionalStencil) {
  runTest(this->getConditionalStencil(), backend, "conditional_stencil.cpp");
}

TEST_F(TestCodeGenNaive, DzCStencil) {
  runTest(this->getStencilFromIIR("update_dz_c"), backend, "update_dz_c.cpp");
}

} // namespace iir
} // namespace dawn
