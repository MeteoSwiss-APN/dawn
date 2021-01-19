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

constexpr auto backend = dawn::codegen::Backend::CUDA;

TEST(Cuda, GlobalIndexStencil) {
  runTest(dawn::getGlobalIndexStencil(), backend, "reference/global_indexing.cu");
}

TEST(Cuda, NonOverlappingInterval) {
  runTest(dawn::getNonOverlappingInterval(), backend, "reference/nonoverlapping_stencil.cu");
}

TEST(Cuda, LaplacianStencil) {
  runTest(dawn::getLaplacianStencil(), backend, "reference/laplacian_stencil.cu");
}

TEST(Cuda, LaplacianStencilNoSync) {
  runTest(dawn::getLaplacianStencil(), backend, "reference/laplacian_stencil_nosync.cu", false);
}

TEST(Cuda, ConditionalStencil) {
  runTest(dawn::IIRSerializer::deserialize("input/conditional_stencil.iir"), backend,
          "reference/conditional_stencil.cu");
}

TEST(Cuda, DzCStencil) {
  runTest(dawn::IIRSerializer::deserialize("input/update_dz_c.iir"), backend,
          "reference/update_dz_c.cu");
}

} // namespace
