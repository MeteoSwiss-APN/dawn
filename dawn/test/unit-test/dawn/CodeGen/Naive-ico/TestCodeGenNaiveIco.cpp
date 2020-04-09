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

class TestCodeGenNaiveIco : public TestCodeGen {};

TEST_F(TestCodeGenNaiveIco, ICONLaplacianStencil) {
  runTest(this->getStencilFromIIR("ICON_laplacian_stencil"), "ICON_laplacian_stencil.cpp");
}

TEST_F(TestCodeGenNaiveIco, SimpleReductionStencil) {
  runTest(this->getStencilFromIIR("simple_reduction_stencil"), "simple_reduction_stencil.cpp");
}

TEST_F(TestCodeGenNaiveIco, SparseDimensionsStencil) {
  runTest(this->getStencilFromIIR("sparse_dimensions"), "sparse_dimensions.cpp");
}

TEST_F(TestCodeGenNaiveIco, TridiagonalSolveUnstructured) {
  runTest(this->getStencilFromIIR("tridiagonal_solve_unstructured"),
          "tridiagonal_solve_unstructured.cpp");
}

} // namespace iir
} // namespace dawn
