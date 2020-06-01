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

#include "dawn/AST/GridType.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/ASTUtil.h"
#include "dawn/IIR/AccessComputation.h"
#include "dawn/IIR/FieldAccessMetadata.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/InstantiationHelper.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassSetStageName.h"
#include "dawn/Optimizer/PassTemporaryType.h"
#include "dawn/Optimizer/StatementMapper.h"
#include "dawn/SIR/ASTFwd.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Support/Logger.h"
#include "dawn/Support/STLExtras.h"

#include <gtest/gtest.h>
#include <memory>
#include <stack>
#include <string>
#include <unistd.h>

#include "GenerateInMemoryStencils.h"

namespace dawn {

TEST(TestIIRDeserializer, CopyStencil) {
  auto instantiation = IIRSerializer::deserialize("reference_iir/copy_stencil.iir");
  EXPECT_EQ(instantiation->getIIR()->getGridType(), ast::GridType::Cartesian);
  EXPECT_EQ(instantiation->getStencils().size(), 1);

  const auto& metaData = instantiation->getMetaData();
  const auto& stencil = instantiation->getStencils()[0];
  EXPECT_EQ(stencil->getChildren().size(), 1);

  auto const& mss = *(stencil->childrenBegin());
  EXPECT_EQ(mss->getChildren().size(), 1);
  EXPECT_EQ(mss->getLoopOrder(), iir::LoopOrderKind::Parallel);

  auto const& stage = *(mss->childrenBegin());
  EXPECT_EQ(stage->getChildren().size(), 1);

  auto const& doMethod = *(stage->childrenBegin());
  auto interval = iir::Interval{0, sir::Interval::End};
  EXPECT_EQ(doMethod->getInterval(), interval);

  auto const& ast = doMethod->getAST();
  EXPECT_EQ(ast.getStatements().size(), 1);

  auto const& stmt = ast.getStatements()[0];
  EXPECT_EQ(stmt->getKind(), ast::Stmt::Kind::ExprStmt);
  const auto& accesses = stmt->getData<iir::IIRStmtData>().CallerAccesses;
  const auto& reads = accesses->getReadAccesses();
  const auto& writes = accesses->getWriteAccesses();

  const auto& reads_it = reads.find(metaData.getAccessIDFromName("in_field"));
  EXPECT_TRUE(reads_it != reads.end());
  const auto& reads_extents = reads_it->second;
  EXPECT_EQ(reads_extents, iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 0, 0));

  const auto& writes_it = writes.find(metaData.getAccessIDFromName("out_field"));
  EXPECT_TRUE(writes_it != writes.end());
  const auto& writes_extents = writes_it->second;
  EXPECT_EQ(writes_extents, iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 0, 0));
}

TEST(TestIIRDeserializer, LapStencil) {
  auto instantiation = IIRSerializer::deserialize("reference_iir/lap_stencil.iir");
  EXPECT_EQ(instantiation->getIIR()->getGridType(), ast::GridType::Cartesian);
  EXPECT_EQ(instantiation->getStencils().size(), 1);

  const auto& metaData = instantiation->getMetaData();
  const auto& stencil = instantiation->getStencils()[0];
  EXPECT_EQ(stencil->getChildren().size(), 1);

  auto const& mss = *(stencil->childrenBegin());
  EXPECT_EQ(mss->getChildren().size(), 2);
  EXPECT_EQ(mss->getLoopOrder(), iir::LoopOrderKind::Parallel);

  auto const& stage_iter = mss->getChildren().begin();
  auto const& stage1 = *(stage_iter);
  EXPECT_EQ(stage1->getChildren().size(), 1);

  auto const& doMethod1 = *(stage1->childrenBegin());
  auto interval = iir::Interval{0, sir::Interval::End};
  EXPECT_EQ(doMethod1->getInterval(), interval);

  auto const& ast1 = doMethod1->getAST();
  EXPECT_EQ(ast1.getStatements().size(), 1);

  auto const& stmt1 = ast1.getStatements()[0];
  EXPECT_EQ(stmt1->getKind(), ast::Stmt::Kind::ExprStmt);
  const auto& accesses1 = stmt1->getData<iir::IIRStmtData>().CallerAccesses;
  const auto& reads1 = accesses1->getReadAccesses();
  const auto& writes1 = accesses1->getWriteAccesses();

  const auto& reads1it = reads1.find(metaData.getAccessIDFromName("in"));
  EXPECT_TRUE(reads1it != reads1.end());
  const auto& reads1_extents = reads1it->second;
  EXPECT_EQ(reads1_extents, iir::Extents(dawn::ast::cartesian, -2, 2, -2, 2, 0, 0));

  const auto& writes1it = writes1.find(metaData.getAccessIDFromName("tmp"));
  EXPECT_TRUE(writes1it != writes1.end());
  const auto& writes1_extents = writes1it->second;
  EXPECT_EQ(writes1_extents, iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 0, 0));

  auto const& stage2 = *(std::next(stage_iter));
  EXPECT_EQ(stage2->getChildren().size(), 1);

  auto const& doMethod2 = *(stage2->childrenBegin());
  interval = iir::Interval{0, sir::Interval::End};
  EXPECT_EQ(doMethod2->getInterval(), interval);

  auto const& ast2 = doMethod2->getAST();
  EXPECT_EQ(ast2.getStatements().size(), 1);

  auto const& stmt2 = ast2.getStatements()[0];
  EXPECT_EQ(stmt2->getKind(), ast::Stmt::Kind::ExprStmt);
  const auto& accesses2 = stmt2->getData<iir::IIRStmtData>().CallerAccesses;
  const auto& reads2 = accesses2->getReadAccesses();
  const auto& writes2 = accesses2->getWriteAccesses();

  const auto& reads2it = reads2.find(metaData.getAccessIDFromName("tmp"));
  EXPECT_TRUE(reads2it != reads2.end());
  const auto& reads2_extents = reads2it->second;
  EXPECT_EQ(reads2_extents, iir::Extents(dawn::ast::cartesian, -1, 1, -1, 1, 0, 0));

  const auto& writes2it = writes2.find(metaData.getAccessIDFromName("out"));
  EXPECT_TRUE(writes2it != writes2.end());
  const auto& writes2_extents = writes2it->second;
  EXPECT_EQ(writes2_extents, iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 0, 0));
}

TEST(TestIIRDeserializer, UnstructuredSumEdgeToCells) {
  auto instantiation =
      IIRSerializer::deserialize("reference_iir/unstructured_sum_edge_to_cells.iir");
  EXPECT_EQ(instantiation->getIIR()->getGridType(), ast::GridType::Unstructured);
  EXPECT_EQ(instantiation->getStencils().size(), 1);

  const auto& metaData = instantiation->getMetaData();
  const auto& stencil = instantiation->getStencils()[0];
  EXPECT_EQ(stencil->getChildren().size(), 1);

  auto const& mss = *(stencil->childrenBegin());
  EXPECT_EQ(mss->getChildren().size(), 2);
  EXPECT_EQ(mss->getLoopOrder(), iir::LoopOrderKind::Parallel);

  auto const& stage_iter = mss->getChildren().begin();
  auto const& stage1 = *(stage_iter);
  EXPECT_EQ(stage1->getLocationType(), ast::LocationType::Edges);
  EXPECT_EQ(stage1->getChildren().size(), 1);

  auto const& doMethod1 = *(stage1->childrenBegin());
  auto interval = iir::Interval{0, sir::Interval::End};
  EXPECT_EQ(doMethod1->getInterval(), interval);

  auto const& ast1 = doMethod1->getAST();
  EXPECT_EQ(ast1.getStatements().size(), 1);

  auto const& stmt1 = ast1.getStatements()[0];
  EXPECT_EQ(stmt1->getKind(), ast::Stmt::Kind::ExprStmt);
  const auto& accesses1 = stmt1->getData<iir::IIRStmtData>().CallerAccesses;
  const auto& reads1 = accesses1->getReadAccesses();
  const auto& writes1 = accesses1->getWriteAccesses();

  std::string literalName = metaData.getNameFromLiteralAccessID(reads1.begin()->first);
  EXPECT_EQ(literalName, "10");

  const auto& writes1it = writes1.find(metaData.getAccessIDFromName("in_field"));
  EXPECT_TRUE(writes1it != writes1.end());
  EXPECT_EQ(writes1it->second, iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 0, 0));

  auto const& stage2 = *(std::next(stage_iter));
  EXPECT_EQ(stage2->getLocationType(), ast::LocationType::Cells);
  EXPECT_EQ(stage2->getChildren().size(), 1);

  auto const& doMethod2 = *(stage2->childrenBegin());
  interval = iir::Interval{0, sir::Interval::End};
  EXPECT_EQ(doMethod2->getInterval(), interval);

  auto const& ast2 = doMethod2->getAST();
  EXPECT_EQ(ast2.getStatements().size(), 1);

  auto const& stmt2 = ast2.getStatements()[0];
  EXPECT_EQ(stmt2->getKind(), ast::Stmt::Kind::ExprStmt);
  const auto& accesses2 = stmt2->getData<iir::IIRStmtData>().CallerAccesses;
  const auto& reads2 = accesses2->getReadAccesses();
  const auto& writes2 = accesses2->getWriteAccesses();

  auto reads2it = reads2.find(-5);
  EXPECT_TRUE(reads2it != reads2.end());
  EXPECT_EQ(metaData.getNameFromLiteralAccessID(reads2it->first), "0.000000");

  reads2it = reads2.find(metaData.getAccessIDFromName("in_field"));
  EXPECT_TRUE(reads2it != reads2.end());
  auto const& reads2_extents =
      iir::extent_cast<iir::UnstructuredExtent const&>(reads2it->second.horizontalExtent());
  EXPECT_TRUE(reads2_extents.hasExtent());

  const auto& writes2it = writes2.find(metaData.getAccessIDFromName("out_field"));
  EXPECT_TRUE(writes2it != writes2.end());
  const auto& writes2_extents = writes2it->second;
  EXPECT_EQ(writes2_extents, iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 0, 0));
}

TEST(TestIIRDeserializer, UnstructuredMixedCopies) {
  auto instantiation = IIRSerializer::deserialize("reference_iir/unstructured_mixed_copies.iir");
  EXPECT_EQ(instantiation->getIIR()->getGridType(), ast::GridType::Unstructured);
  EXPECT_EQ(instantiation->getStencils().size(), 1);

  const auto& metaData = instantiation->getMetaData();
  const auto& stencil = instantiation->getStencils()[0];
  EXPECT_EQ(stencil->getChildren().size(), 1);

  auto const& mss = *(stencil->childrenBegin());
  EXPECT_EQ(mss->getChildren().size(), 2);
  EXPECT_EQ(mss->getLoopOrder(), iir::LoopOrderKind::Forward);

  auto const& stage_iter = mss->getChildren().begin();
  auto const& stage1 = *(stage_iter);
  EXPECT_EQ(stage1->getChildren().size(), 1);

  auto const& doMethod1 = *(stage1->childrenBegin());
  auto interval = iir::Interval{0, sir::Interval::End};
  EXPECT_EQ(doMethod1->getInterval(), interval);

  auto const& ast1 = doMethod1->getAST();
  EXPECT_EQ(ast1.getStatements().size(), 1);

  auto const& stmt1 = ast1.getStatements()[0];
  EXPECT_EQ(stmt1->getKind(), ast::Stmt::Kind::ExprStmt);
  const auto& accesses1 = stmt1->getData<iir::IIRStmtData>().CallerAccesses;
  const auto& reads1 = accesses1->getReadAccesses();
  const auto& writes1 = accesses1->getWriteAccesses();

  const auto& reads1it = reads1.find(metaData.getAccessIDFromName("in_c"));
  EXPECT_TRUE(reads1it != reads1.end());
  EXPECT_EQ(reads1it->second, iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 0, 0));

  const auto& writes1it = writes1.find(metaData.getAccessIDFromName("out_c"));
  EXPECT_TRUE(writes1it != writes1.end());
  EXPECT_EQ(writes1it->second, iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 0, 0));

  auto const& stage2 = *(std::next(stage_iter));
  EXPECT_EQ(stage2->getChildren().size(), 1);

  auto const& doMethod2 = *(stage2->childrenBegin());
  interval = iir::Interval{0, sir::Interval::End};
  EXPECT_EQ(doMethod2->getInterval(), interval);

  auto const& ast2 = doMethod2->getAST();
  EXPECT_EQ(ast2.getStatements().size(), 1);

  auto const& stmt2 = ast2.getStatements()[0];
  EXPECT_EQ(stmt2->getKind(), ast::Stmt::Kind::ExprStmt);
  const auto& accesses2 = stmt2->getData<iir::IIRStmtData>().CallerAccesses;
  const auto& reads2 = accesses2->getReadAccesses();
  const auto& writes2 = accesses2->getWriteAccesses();

  const auto& reads2it = reads2.find(metaData.getAccessIDFromName("in_e"));
  EXPECT_TRUE(reads2it != reads2.end());
  EXPECT_EQ(reads2it->second, iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 0, 0));

  const auto& writes2it = writes2.find(metaData.getAccessIDFromName("out_e"));
  EXPECT_TRUE(writes2it != writes2.end());
  EXPECT_EQ(writes2it->second, iir::Extents(dawn::ast::cartesian, 0, 0, 0, 0, 0, 0));
}

} // namespace dawn
