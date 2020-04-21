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

constexpr auto backend = dawn::codegen::Backend::CXXNaive;

TEST(TestCodeGenNaive, GlobalIndexStencil) {
  runTest(dawn::getGlobalIndexStencil(), backend, "reference/global_indexing.cpp");
}

TEST(TestCodeGenNaive, NonOverlappingInterval) {
  runTest(dawn::getNonOverlappingInterval(), backend, "reference/nonoverlapping_stencil.cpp");
}

TEST(TestCodeGenNaive, LaplacianStencil) {
  runTest(dawn::getLaplacianStencil(), backend, "reference/laplacian_stencil.cpp");
}

TEST(TestCodeGenNaive, ConditionalStencil) {
  runTest(dawn::IIRSerializer::deserialize("input/conditional_stencil.iir"), backend,
          "reference/conditional_stencil.cpp");
}

TEST(TestCodeGenNaive, DzCStencil) {
  runTest(dawn::IIRSerializer::deserialize("input/update_dz_c.iir"), backend,
          "reference/update_dz_c.cpp");
}

} // namespace
