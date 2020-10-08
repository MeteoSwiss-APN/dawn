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

#include "UnstructuredStencils.h"
#include "dawn/CodeGen/Cuda-ico/IcoChainSizes.h"
#include "dawn/CodeGen/Options.h"
#include "dawn/Serialization/IIRSerializer.h"

#include <gtest/gtest.h>

namespace {

constexpr auto backend = dawn::codegen::Backend::CUDAIco;

// NOTE: Often-changing backend. For the moment we prefer to test code generation through end-to-end
// tests checking the output. To be reconsidered once this is stable.

TEST(CudaIco, ChainSizes) {
  std::map<std::vector<dawn::ast::LocationType>, int> tests{
      {{dawn::ast::LocationType::Cells, dawn::ast::LocationType::Edges}, 3},
      {{dawn::ast::LocationType::Cells, dawn::ast::LocationType::Vertices}, 3},
      {{dawn::ast::LocationType::Edges, dawn::ast::LocationType::Cells}, 2},
      {{dawn::ast::LocationType::Edges, dawn::ast::LocationType::Vertices}, 2},
      {{dawn::ast::LocationType::Vertices, dawn::ast::LocationType::Cells}, 6},
      {{dawn::ast::LocationType::Vertices, dawn::ast::LocationType::Edges}, 6},
      {{dawn::ast::LocationType::Edges, dawn::ast::LocationType::Cells,
        dawn::ast::LocationType::Edges},
       4},
      {{dawn::ast::LocationType::Edges, dawn::ast::LocationType::Cells,
        dawn::ast::LocationType::Vertices},
       4},
      {{dawn::ast::LocationType::Edges, dawn::ast::LocationType::Cells,
        dawn::ast::LocationType::Vertices, dawn::ast::LocationType::Cells},
       16},
      {{dawn::ast::LocationType::Cells, dawn::ast::LocationType::Vertices,
        dawn::ast::LocationType::Cells},
       12},
      {{dawn::ast::LocationType::Cells, dawn::ast::LocationType::Vertices,
        dawn::ast::LocationType::Cells, dawn::ast::LocationType::Edges},
       24},
      {{dawn::ast::LocationType::Edges, dawn::ast::LocationType::Vertices,
        dawn::ast::LocationType::Edges},
       10},
      {{dawn::ast::LocationType::Edges, dawn::ast::LocationType::Vertices,
        dawn::ast::LocationType::Edges, dawn::ast::LocationType::Cells},
       10},
      {{dawn::ast::LocationType::Vertices, dawn::ast::LocationType::Edges,
        dawn::ast::LocationType::Cells},
       6},
      {{dawn::ast::LocationType::Vertices, dawn::ast::LocationType::Edges,
        dawn::ast::LocationType::Cells, dawn::ast::LocationType::Vertices},
       6},
      {{dawn::ast::LocationType::Vertices, dawn::ast::LocationType::Edges,
        dawn::ast::LocationType::Cells, dawn::ast::LocationType::Vertices,
        dawn::ast::LocationType::Edges},
       30},
      {{dawn::ast::LocationType::Vertices, dawn::ast::LocationType::Edges,
        dawn::ast::LocationType::Cells, dawn::ast::LocationType::Vertices,
        dawn::ast::LocationType::Edges, dawn::ast::LocationType::Cells},
       24},
      {{dawn::ast::LocationType::Cells, dawn::ast::LocationType::Edges,
        dawn::ast::LocationType::Cells},
       3},
      {{dawn::ast::LocationType::Cells, dawn::ast::LocationType::Edges,
        dawn::ast::LocationType::Cells, dawn::ast::LocationType::Edges},
       9},
      {{dawn::ast::LocationType::Cells, dawn::ast::LocationType::Edges,
        dawn::ast::LocationType::Cells, dawn::ast::LocationType::Edges,
        dawn::ast::LocationType::Cells},
       9},
      {{dawn::ast::LocationType::Cells, dawn::ast::LocationType::Edges,
        dawn::ast::LocationType::Cells, dawn::ast::LocationType::Edges,
        dawn::ast::LocationType::Cells, dawn::ast::LocationType::Edges},
       21},
  };

  for(const auto& [chain, expected] : tests) {
    const auto actual = dawn::ICOChainSizesComputed(chain);
    EXPECT_EQ(expected, actual);
  }
}

} // namespace