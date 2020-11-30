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
#include "dawn/IIR/ASTStmt.h"
#include "dawn/Unittest/IIRBuilder.h"
#include "dawn/Validator/IndirectionChecker.h"

#include <gtest/gtest.h>
#include <memory>

using namespace dawn;

namespace {
TEST(IndirectionCheckerTest, Case_Pass) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto in = b.field("in", LocType::Cells);
  auto out = b.field("out", LocType::Cells);
  auto kidx = b.field("kidx", LocType::Cells);

  auto stencil = b.build(
      "pass", b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(
                      dawn::sir::Interval::Start, dawn::sir::Interval::End,
                      b.stmt(b.assignExpr(
                          b.at(out), b.at(in, AccessType::r,
                                          ast::Offsets{ast::unstructured, false, 1, "kidx"}))))))));

  auto result = IndirectionChecker::checkIndirections(*stencil->getIIR());
  EXPECT_EQ(result, IndirectionChecker::IndirectionResult(true, dawn::SourceLocation()));
}

TEST(IndirectionCheckerTest, Case_Fail) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto in = b.field("in", LocType::Cells);
  auto out = b.field("out", LocType::Cells);
  auto kidx = b.field("kidx", LocType::Cells);

  auto stencil = b.build(
      "fail", b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(
                      dawn::sir::Interval::Start, dawn::sir::Interval::End,
                      b.stmt(b.assignExpr(
                          b.at(out), b.at(in, AccessType::r,
                                          ast::Offsets{ast::unstructured, false, 1, "kidx"}))))))));

  // inject an indirected read into the offset of the indirected read
  //  out[c,k] = in[kidx[kidx[c,k]]]
  // which is prohibited
  for(auto stmt : dawn::iterateIIROverStmt(*stencil->getIIR())) {
    if(auto exprStmt = dyn_pointer_cast<ExprStmt>(stmt)) {
      if(auto assignExpr = dyn_pointer_cast<AssignmentExpr>(exprStmt->getExpr())) {
        auto rhs = dyn_pointer_cast<FieldAccessExpr>(assignExpr->getRight());
        std::dynamic_pointer_cast<FieldAccessExpr>(
            rhs->getOffset().getVerticalIndirectionFieldAsExpr())
            ->getOffset()
            .setVerticalIndirection("kidx");
      }
    }
  }

  auto result = IndirectionChecker::checkIndirections(*stencil->getIIR());
  EXPECT_EQ(result, IndirectionChecker::IndirectionResult(false, dawn::SourceLocation()));
}

TEST(IndirectionCheckerTest, Case_Fail2) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto in = b.field("in", LocType::Cells);
  auto out = b.field("out", LocType::Cells);
  auto kidx = b.field("kidx", LocType::Cells);

  // vertically indirected _write_, which is prohibited!
  auto stencil = b.build(
      "fail", b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(
                      dawn::sir::Interval::Start, dawn::sir::Interval::End,
                      b.stmt(b.assignExpr(b.at(out, AccessType::rw,
                                               ast::Offsets{ast::unstructured, false, 1, "kidx"}),
                                          b.at(in))))))));

  auto result = IndirectionChecker::checkIndirections(*stencil->getIIR());
  EXPECT_EQ(result, IndirectionChecker::IndirectionResult(false, dawn::SourceLocation()));
}

TEST(IndirectionCheckerTest, Case_Fail3) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto in = b.field("in", LocType::Cells);
  auto out = b.field("out", LocType::Cells);

  // vertically indirected _write_, which is prohibited!
  auto stencil = b.build(
      "fail",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                             b.stmt(b.assignExpr(b.at(out, AccessType::rw,
                                                      ast::Offsets{ast::unstructured, false, 1}),
                                                 b.at(in))))))));

  auto result = IndirectionChecker::checkIndirections(*stencil->getIIR());
  EXPECT_EQ(result, IndirectionChecker::IndirectionResult(false, dawn::SourceLocation()));
}

} // namespace