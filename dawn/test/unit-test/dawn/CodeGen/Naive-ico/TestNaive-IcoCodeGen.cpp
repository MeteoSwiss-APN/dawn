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

#include "Stencils.h"
#include "dawn/CodeGen/Options.h"
#include "dawn/Serialization/IIRSerializer.h"

#include <gtest/gtest.h>

namespace {

constexpr auto backend = dawn::codegen::Backend::CXXNaiveIco;

TEST(NaiveIco, ICONLaplacianStencil) {
  runTest(dawn::IIRSerializer::deserialize("input/ICON_laplacian_stencil.iir"), backend,
          "reference/ICON_laplacian_stencil.cpp");
}

TEST(NaiveIco, SimpleReductionStencil) {
  runTest(dawn::IIRSerializer::deserialize("input/simple_reduction_stencil.iir"), backend,
          "reference/simple_reduction_stencil.cpp");
}

TEST(NaiveIco, SparseDimensionsStencil) {
  runTest(dawn::IIRSerializer::deserialize("input/sparse_dimensions.iir"), backend,
          "reference/sparse_dimensions.cpp");
}

TEST(NaiveIco, TridiagonalSolveUnstructured) {
  runTest(dawn::IIRSerializer::deserialize("input/tridiagonal_solve_unstructured.iir"), backend,
          "reference/tridiagonal_solve_unstructured.cpp");
}

} // namespace
