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

#include "dawn/AST/Offsets.h"
#include "dawn/Unittest/IIRBuilder.h"
#include "dawn/Validator/IndirectionChecker.h"

#include <gtest/gtest.h>
#include <memory>

using namespace dawn;

namespace {
TEST(IndirectionCheckerTest, Case_1) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto in = b.field("in", LocType::Cells);
  auto out = b.field("out", LocType::Cells);
  auto kidx = b.field("kidx", LocType::Cells);

  b.build("pass", b.stencil(b.multistage(
                      LoopOrderKind::Parallel,
                      b.stage(b.doMethod(
                          dawn::sir::Interval::Start, dawn::sir::Interval::End,
                          b.stmt(b.assignExpr(b.at(in), b.at(out, AccessType::r,
                                                             ast::Offsets{ast::unstructured, true,
                                                                          1, "kidx"}))))))));

  auto result = IndirectionChecker::checkIndirections(*stencil->getIIR(), stencil->getMetaData());
  EXPECT_EQ(result, IndirectionChecker::indirectionResult(true, dawn::SourceLocation()));
}

} // namespace