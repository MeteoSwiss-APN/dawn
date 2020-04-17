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
#include "dawn/Compiler/DawnCompiler.h"
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
#include "dawn/Support/Logging.h"
#include "dawn/Support/STLExtras.h"

#include <gtest/gtest.h>
#include <memory>
#include <stack>
#include <string>
#include <unistd.h>

#include "GenerateInMemoryStencils.h"

namespace dawn {

TEST(TestIIRDeserializer, CopyStencil) {
  // generate IIR in memory
  UIDGenerator::getInstance()->reset();
  auto instantiation = IIRSerializer::deserialize("reference_iir/copy_stencil.iir");
  EXPECT_EQ(instantiation->getStencils().size(), 1);

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
}

TEST(TestIIRDeserializer, LapStencil) {
  // generate IIR in memory
  UIDGenerator::getInstance()->reset();
  auto instantiation = IIRSerializer::deserialize("reference_iir/lap_stencil.iir");
  EXPECT_EQ(instantiation->getStencils().size(), 1);

  const auto& stencil = instantiation->getStencils()[0];
  EXPECT_EQ(stencil->getChildren().size(), 1);

  auto const& mss = *(stencil->childrenBegin());
  EXPECT_EQ(mss->getChildren().size(), 2);
  EXPECT_EQ(mss->getLoopOrder(), iir::LoopOrderKind::Parallel);

  auto const& stage_iter = mss->getChildren().begin();
  auto const& stage1 = *(stage_iter);
  EXPECT_EQ(stage1->getChildren().size(), 1);

  auto const& doMethod = *(stage1->childrenBegin());
  auto interval = iir::Interval{0, sir::Interval::End};
  EXPECT_EQ(doMethod->getInterval(), interval);

  auto const& ast = doMethod->getAST();
  EXPECT_EQ(ast.getStatements().size(), 1);

  auto const& stage2 = *(std::next(stage_iter));
  EXPECT_EQ(stage2->getChildren().size(), 1);

  auto const& doMethod2 = *(stage2->childrenBegin());
  interval = iir::Interval{0, sir::Interval::End};
  EXPECT_EQ(doMethod2->getInterval(), interval);

  auto const& ast2 = doMethod2->getAST();
  EXPECT_EQ(ast2.getStatements().size(), 1);
}

TEST(TestIIRDeserializer, UnstructuredMixedCopies) {
  // generate IIR in memory
  UIDGenerator::getInstance()->reset();
  auto instantiation = IIRSerializer::deserialize("reference_iir/unstructured_mixed_copies.iir");
  EXPECT_EQ(instantiation->getStencils().size(), 1);

  const auto& stencil = instantiation->getStencils()[0];
  EXPECT_EQ(stencil->getChildren().size(), 1);

  auto const& mss = *(stencil->childrenBegin());
  EXPECT_EQ(mss->getChildren().size(), 2);
  EXPECT_EQ(mss->getLoopOrder(), iir::LoopOrderKind::Forward);

  auto const& stage_iter = mss->getChildren().begin();
  auto const& stage1 = *(stage_iter);
  EXPECT_EQ(stage1->getChildren().size(), 1);

  auto const& doMethod = *(stage1->childrenBegin());
  auto interval = iir::Interval{0, sir::Interval::End};
  EXPECT_EQ(doMethod->getInterval(), interval);

  auto const& ast = doMethod->getAST();
  EXPECT_EQ(ast.getStatements().size(), 1);

  auto const& stage2 = *(std::next(stage_iter));
  EXPECT_EQ(stage2->getChildren().size(), 1);

  auto const& doMethod2 = *(stage2->childrenBegin());
  interval = iir::Interval{0, sir::Interval::End};
  EXPECT_EQ(doMethod2->getInterval(), interval);

  auto const& ast2 = doMethod2->getAST();
  EXPECT_EQ(ast2.getStatements().size(), 1);
}

} // anonymous namespace
