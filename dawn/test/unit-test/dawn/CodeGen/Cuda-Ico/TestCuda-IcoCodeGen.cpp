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
#include "dawn/CodeGen/Cuda-ico/LocToStringUtils.h"
#include "dawn/CodeGen/Cuda-ico/ReductionMerger.h"
#include "dawn/CodeGen/IcoChainSizes.h"
#include "dawn/CodeGen/Options.h"
#include "dawn/Serialization/IIRSerializer.h"

#include <gtest/gtest.h>

namespace {

constexpr auto backend = dawn::codegen::Backend::CUDAIco;

// NOTE: Often-changing backend. For the moment we prefer to test code generation through end-to-end
// tests checking the output. To be reconsidered once this is stable.

TEST(CudaIco, ChainSizes) {
  std::map<dawn::ast::NeighborChain, int> tests{
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
    const auto actual = dawn::ICOChainSize(chain);
    EXPECT_EQ(expected, actual) << "Incorrect neighbor size: " << actual << " (expected "
                                << expected << ") for neighborchain: "
                                << dawn::codegen::cudaico::chainToVectorString(chain);
  }
}

TEST(CudaIco, ReductionMerger_0) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto a_f = b.field("a", dawn::ast::LocationType::Cells);
  auto b_f = b.field("b", dawn::ast::LocationType::Edges);
  auto c_f = b.field("c", dawn::ast::LocationType::Cells);
  auto d_f = b.field("d", dawn::ast::LocationType::Edges);

  // most basic case that can be merged
  auto stencil = b.build(
      "merge",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(dawn::ast::LocationType::Cells,
                  b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                             b.stmt(b.assignExpr(b.at(a_f), b.reduceOverNeighborExpr(
                                                                Op::plus, b.at(b_f), b.lit(0.),
                                                                {dawn::ast::LocationType::Cells,
                                                                 dawn::ast::LocationType::Edges}))),
                             b.stmt(b.assignExpr(
                                 b.at(c_f),
                                 b.reduceOverNeighborExpr(Op::plus, b.at(d_f), b.lit(0.),
                                                          {dawn::ast::LocationType::Cells,
                                                           dawn::ast::LocationType::Edges}))))))));

  auto mergeGroupMap =
      dawn::codegen::cudaico::ReductionMergeGroupsComputer::ComputeReductionMergeGroups(stencil);

  EXPECT_EQ(mergeGroupMap.size(), 1);
  auto mergeGroup = mergeGroupMap.begin()->second;
  EXPECT_EQ(mergeGroup.size(), 1);
  EXPECT_EQ(mergeGroup[0].size(), 2);
}

TEST(CudaIco, ReductionMerger_1) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto a_f = b.field("a", dawn::ast::LocationType::Cells);
  auto b_f = b.field("b", dawn::ast::LocationType::Edges);
  auto c_f = b.field("c", dawn::ast::LocationType::Cells);
  auto d_f = b.field("d", dawn::ast::LocationType::Vertices);

  // most basic case that can't be merged
  auto stencil = b.build(
      "dont_merge",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(
              dawn::ast::LocationType::Cells,
              b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                         b.stmt(b.assignExpr(b.at(a_f), b.reduceOverNeighborExpr(
                                                            Op::plus, b.at(b_f), b.lit(0.),
                                                            {dawn::ast::LocationType::Cells,
                                                             dawn::ast::LocationType::Edges}))),
                         b.stmt(b.assignExpr(
                             b.at(c_f),
                             b.reduceOverNeighborExpr(Op::plus, b.at(d_f), b.lit(0.),
                                                      {dawn::ast::LocationType::Cells,
                                                       dawn::ast::LocationType::Vertices}))))))));

  auto mergeGroupMap =
      dawn::codegen::cudaico::ReductionMergeGroupsComputer::ComputeReductionMergeGroups(stencil);

  EXPECT_EQ(mergeGroupMap.size(), 1);
  auto mergeGroup = mergeGroupMap.begin()->second;
  EXPECT_EQ(mergeGroup.size(), 2);
  EXPECT_EQ(mergeGroup[0].size(), 1);
  EXPECT_EQ(mergeGroup[1].size(), 1);
}

TEST(CudaIco, ReductionMerger_2) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto a_f = b.field("a", dawn::ast::LocationType::Cells);
  auto b_f = b.field("b", dawn::ast::LocationType::Cells);
  auto c_f = b.field("c", dawn::ast::LocationType::Cells);

  // case with a data dependency that can't be merged
  auto stencil = b.build(
      "dont_merge",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(
              dawn::ast::LocationType::Cells,
              b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                         b.stmt(b.assignExpr(b.at(a_f), b.reduceOverNeighborExpr(
                                                            Op::plus, b.at(b_f), b.lit(0.),
                                                            {dawn::ast::LocationType::Cells,
                                                             dawn::ast::LocationType::Edges,
                                                             dawn::ast::LocationType::Cells}))),
                         b.stmt(b.assignExpr(
                             b.at(c_f),
                             b.reduceOverNeighborExpr(
                                 Op::plus, b.at(a_f, HOffsetType::noOffset, 0), b.lit(0.),
                                 {dawn::ast::LocationType::Cells, dawn::ast::LocationType::Edges,
                                  dawn::ast::LocationType::Cells}))))))));

  auto mergeGroupMap =
      dawn::codegen::cudaico::ReductionMergeGroupsComputer::ComputeReductionMergeGroups(stencil);

  EXPECT_EQ(mergeGroupMap.size(), 1);
  auto mergeGroup = mergeGroupMap.begin()->second;
  EXPECT_EQ(mergeGroup.size(), 2);
  EXPECT_EQ(mergeGroup[0].size(), 1);
  EXPECT_EQ(mergeGroup[1].size(), 1);
}

TEST(CudaIco, ReductionMerger_3) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto a_f = b.field("a", dawn::ast::LocationType::Cells);
  auto b_f = b.field("b", dawn::ast::LocationType::Edges);
  auto c_f = b.field("c", dawn::ast::LocationType::Cells);
  auto d_f = b.field("d", dawn::ast::LocationType::Edges);
  auto e_f = b.field("e", dawn::ast::LocationType::Cells);

  // a statement in between to mege-able statements prevents the merge
  //    (possibly to be lifted in the future)
  auto stencil = b.build(
      "merge",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(dawn::ast::LocationType::Cells,
                  b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                             b.stmt(b.assignExpr(b.at(a_f), b.reduceOverNeighborExpr(
                                                                Op::plus, b.at(b_f), b.lit(0.),
                                                                {dawn::ast::LocationType::Cells,
                                                                 dawn::ast::LocationType::Edges}))),
                             b.stmt(b.assignExpr(b.at(e_f), b.lit(1.))),
                             b.stmt(b.assignExpr(
                                 b.at(c_f),
                                 b.reduceOverNeighborExpr(Op::plus, b.at(d_f), b.lit(0.),
                                                          {dawn::ast::LocationType::Cells,
                                                           dawn::ast::LocationType::Edges}))))))));

  auto mergeGroupMap =
      dawn::codegen::cudaico::ReductionMergeGroupsComputer::ComputeReductionMergeGroups(stencil);

  EXPECT_EQ(mergeGroupMap.size(), 1);
  auto mergeGroup = mergeGroupMap.begin()->second;
  EXPECT_EQ(mergeGroup.size(), 2);
  EXPECT_EQ(mergeGroup[0].size(), 1);
  EXPECT_EQ(mergeGroup[1].size(), 1);
}

TEST(CudaIco, ReductionMerger_4) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto a_f = b.field("a", dawn::ast::LocationType::Cells);
  auto b_f = b.field("b", dawn::ast::LocationType::Edges);
  auto c_f = b.field("c", dawn::ast::LocationType::Cells);
  auto d_f = b.field("d", dawn::ast::LocationType::Edges);
  auto e_f = b.field("e", dawn::ast::LocationType::Cells);
  auto f_f = b.field("f", dawn::ast::LocationType::Edges);
  auto g_f = b.field("g", dawn::ast::LocationType::Cells);
  auto h_f = b.field("h", dawn::ast::LocationType::Edges);
  auto i_f = b.field("i", dawn::ast::LocationType::Cells);
  auto j_f = b.field("j", dawn::ast::LocationType::Edges);

  auto mask_f = b.field("mask", dawn::ast::LocationType::Cells);

  auto ifBlock = b.block(
      b.stmt(b.assignExpr(b.at(c_f), b.reduceOverNeighborExpr(Op::plus, b.at(d_f), b.lit(0.),
                                                              {dawn::ast::LocationType::Cells,
                                                               dawn::ast::LocationType::Edges}))),
      b.stmt(b.assignExpr(b.at(i_f), b.reduceOverNeighborExpr(Op::plus, b.at(j_f), b.lit(0.),
                                                              {dawn::ast::LocationType::Cells,
                                                               dawn::ast::LocationType::Edges}))));

  auto elseBlock = b.block(
      b.stmt(b.assignExpr(b.at(e_f), b.reduceOverNeighborExpr(Op::plus, b.at(f_f), b.lit(0.),
                                                              {dawn::ast::LocationType::Cells,
                                                               dawn::ast::LocationType::Edges}))));

  int ifBlockID = ifBlock->getID();
  int elseBlockID = elseBlock->getID();

  // do not merge across boundary of scopes, but merge inside of scopes
  auto stencil = b.build(
      "merge_some",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(dawn::ast::LocationType::Cells,
                  b.doMethod(dawn::ast::Interval::Start, dawn::ast::Interval::End,
                             b.stmt(b.assignExpr(b.at(a_f), b.reduceOverNeighborExpr(
                                                                Op::plus, b.at(b_f), b.lit(0.),
                                                                {dawn::ast::LocationType::Cells,
                                                                 dawn::ast::LocationType::Edges}))),
                             b.ifStmt(b.at(mask_f), std::move(ifBlock), std::move(elseBlock)),
                             b.stmt(b.assignExpr(
                                 b.at(g_f),
                                 b.reduceOverNeighborExpr(Op::plus, b.at(h_f), b.lit(0.),
                                                          {dawn::ast::LocationType::Cells,
                                                           dawn::ast::LocationType::Edges}))))))));

  auto mergeGroupMap =
      dawn::codegen::cudaico::ReductionMergeGroupsComputer::ComputeReductionMergeGroups(stencil);

  for(auto block : mergeGroupMap) {
    if(block.first == ifBlockID) {
      // merged reduction in "if" block
      EXPECT_EQ(block.second.size(), 1);
      EXPECT_EQ(block.second[0].size(), 2);
    } else if(block.first == elseBlockID) {
      // exactly one reduciton in "else" block
      EXPECT_EQ(block.second.size(), 1);
      EXPECT_EQ(block.second[0].size(), 1);
    } else { // two unmerged reduction on "root" scope/block
      EXPECT_EQ(block.second.size(), 2);
      EXPECT_EQ(block.second[0].size(), 1);
      EXPECT_EQ(block.second[1].size(), 1);
    }
  }
}

TEST(CudaIco, ReductionMerger_5) {
  using namespace dawn::iir;

  UnstructuredIIRBuilder b;
  auto a_f = b.field("a", dawn::ast::LocationType::Cells);
  auto b_f = b.field("b", dawn::ast::LocationType::Edges);
  auto c_f = b.field("c", dawn::ast::LocationType::Edges);

  // merge correctly even if the two reductions are on the same line
  auto stencil = b.build(
      "merge",
      b.stencil(b.multistage(
          dawn::iir::LoopOrderKind::Forward,
          b.stage(dawn::ast::LocationType::Cells,
                  b.doMethod(
                      dawn::ast::Interval::Start, dawn::ast::Interval::End,
                      b.stmt(b.assignExpr(
                          b.at(a_f),
                          b.binaryExpr(b.reduceOverNeighborExpr(Op::plus, b.at(b_f), b.lit(0.),
                                                                {dawn::ast::LocationType::Cells,
                                                                 dawn::ast::LocationType::Edges}),
                                       b.reduceOverNeighborExpr(Op::plus, b.at(c_f), b.lit(0.),
                                                                {dawn::ast::LocationType::Cells,
                                                                 dawn::ast::LocationType::Edges}),
                                       Op::plus))))))));

  auto mergeGroupMap =
      dawn::codegen::cudaico::ReductionMergeGroupsComputer::ComputeReductionMergeGroups(stencil);

  EXPECT_EQ(mergeGroupMap.size(), 1);
  auto mergeGroup = mergeGroupMap.begin()->second;
  EXPECT_EQ(mergeGroup.size(), 1);
  EXPECT_EQ(mergeGroup[0].size(), 2);
}

} // namespace