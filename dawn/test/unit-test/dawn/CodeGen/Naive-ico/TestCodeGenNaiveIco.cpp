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

class TestCodeGenNaiveIco : public TestCodeGen {};

TEST_F(TestCodeGenNaiveIco, ICONLaplacianStencil) {
  runTest(this->getStencilFromIIRFile("../input/ICON_laplacian_stencil.iir"),
          "ICON_laplacian_stencil.cpp");
}

TEST_F(TestCodeGenNaiveIco, SimpleReductionStencil) {
  runTest(this->getStencilFromIIRFile("../input/simple_reduction_stencil.iir"),
          "simple_reduction_stencil.cpp");
}

TEST_F(TestCodeGenNaiveIco, SparseDimensionsStencil) {
  runTest(this->getStencilFromIIRFile("../input/sparse_dimensions.iir"), "sparse_dimensions.cpp");
}

TEST_F(TestCodeGenNaiveIco, TridiagonalSolveUnstructured) {
  runTest(this->getStencilFromIIRFile("../input/tridiagonal_solve_unstructured.iir"),
          "tridiagonal_solve_unstructured.cpp");
}

} // anonymous namespace
