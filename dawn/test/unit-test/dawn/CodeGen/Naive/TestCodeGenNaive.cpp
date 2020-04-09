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

class TestCodeGenNaive : public TestCodeGen {};

TEST_F(TestCodeGenNaive, CopyStencil) {
  runTest(this->getStencilFromIIR("copy_stencil"), "copy_stencil.cpp");
}

TEST_F(TestCodeGenNaive, GlobalIndexStencilFromFile) {
  runTest(this->getStencilFromIIR("global_index_stencil"), "global_index_stencil.cpp");
}

TEST_F(TestCodeGenNaive, HoriDiffStencil) {
  runTest(this->getStencilFromIIR("hori_diff_stencil"), "hori_diff_stencil.cpp");
}

TEST_F(TestCodeGenNaive, TridiagonalSolveStencil) {
  runTest(this->getStencilFromIIR("tridiagonal_solve_stencil"), "tridiagonal_solve_stencil.cpp");
}

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
  runTest(this->getStencilFromIIR("conditional_stencil"), "conditional_stencil.cpp");
}

TEST_F(TestCodeGenNaive, DzCStencil) {
  runTest(this->getStencilFromIIR("update_dz_c"), "update_dz_c.cpp");
}

} // namespace iir
} // namespace dawn
