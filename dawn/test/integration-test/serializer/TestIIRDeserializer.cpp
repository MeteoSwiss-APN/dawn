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
  auto copy_stencil_memory = IIRSerializer::deserialize("reference_iir/copy_stencil.iir");
  EXPECT_EQ(copy_stencil_memory->getStencils().size(), 1);

  const auto& stencil = copy_stencil_memory->getStencils()[0];
  EXPECT_EQ(stencil->getChildren().size(), 1);

  auto const& mss = *(stencil->childrenBegin());
  EXPECT_EQ(mss->getChildren().size(), 1);
  EXPECT_EQ(mss->getLoopOrder(), iir::LoopOrderKind::Parallel);

  auto const& stage = *(mss->childrenBegin());
  EXPECT_EQ(stage->getChildren().size(), 1);

  auto const& doMethod = *(stage->childrenBegin());
  auto interval = iir::Interval{0, sir::Interval::End};
  EXPECT_EQ(doMethod->getInterval(), interval);

  // Assert something about AST...
}

} // anonymous namespace
