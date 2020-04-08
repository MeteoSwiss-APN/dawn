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

class TestCodeGenCuda : public TestCodeGen {};

TEST_F(TestCodeGenCuda, CopyStencil) {
  runTest(this->getStencilFromIIRFile("../input/copy_stencil.iir"), "copy_stencil.cu");
}

TEST_F(TestCodeGenCuda, GlobalIndexStencilFromFile) {
  runTest(this->getStencilFromIIRFile("../input/global_index_stencil.iir"),
          "global_index_stencil.cu");
}

TEST_F(TestCodeGenCuda, HoriDiffStencil) {
  runTest(this->getStencilFromIIRFile("../input/hori_diff_stencil.iir"), "hori_diff_stencil.cu");
}

TEST_F(TestCodeGenCuda, TridiagonalSolveStencil) {
  runTest(this->getStencilFromIIRFile("../input/tridiagonal_solve_stencil.iir"),
          "tridiagonal_solve_stencil.cu");
}

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
  runTest(this->getStencilFromIIRFile("../input/conditional_stencil.iir"),
          "conditional_stencil.cu");
}

TEST_F(TestCodeGenCuda, DzCStencil) {
  runTest(this->getStencilFromIIR("update_dz_c"), "update_dz_c.cu");
}

} // namespace iir
} // namespace dawn
