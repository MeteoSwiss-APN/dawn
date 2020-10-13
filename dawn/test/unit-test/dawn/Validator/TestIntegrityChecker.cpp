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

#include "dawn/IIR/IIR.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Support/Exception.h"
#include "dawn/Unittest/IIRBuilder.h"
#include "dawn/Validator/IntegrityChecker.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

TEST(TestIntegrityChecker, GlobalsOptimizedAway) {
  // Load IIR from file
  auto instantiation = IIRSerializer::deserialize("input/globals_opt_away.iir");
  IntegrityChecker checker(instantiation.get());

  // Run integrity check and succeed if exception is thrown
  try {
    checker.run();
    FAIL() << "Semantic error not thrown";
  } catch(SemanticError& error) {
    SUCCEED();
  }
}
TEST(TestIntegrityChecker, OffsetReadsInCorrectContext) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto in = b.field("in", LocType::Cells);
  auto out = b.field("out", LocType::Cells);

  try {
    auto stencil = b.build(
        "OffsetReadsInCorrectContext",
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                               b.stmt(b.assignExpr(
                                   b.at(out), b.at(in, AccessType::r,
                                                   ast::Offsets{ast::unstructured, true, 0}))))))));
    FAIL() << "Semantic error not thrown";
  } catch(SemanticError& error) {
    SUCCEED();
  }
}
TEST(TestIntegrityChecker, Assignment2d3d) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto in = b.field("in", LocType::Edges, true);
  auto out = b.field("out", LocType::Edges, false);

  try {
    auto stencil = b.build(
        "Assignment2d3d",
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(LocType::Edges, b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                               b.stmt(b.assignExpr(b.at(out), b.at(in))))))));
    FAIL() << "Semantic error not thrown";
  } catch(SemanticError& error) {
    SUCCEED();
  }
}
TEST(TestIntegrityChecker, Assignment1d3d) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;

  auto f_e = b.field("edges", LocType::Edges);
  auto f_vert = b.vertical_field("vert");

  try {
    auto stencil = b.build(
        "Assignment1d3d",
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(LocType::Edges, b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                               b.stmt(b.assignExpr(b.at(f_vert), b.at(f_e))))))));
    FAIL() << "Semantic error not thrown";
  } catch(SemanticError& error) {
    SUCCEED();
  }
}

TEST(TestIntegrityChecker, OffsetReadsIn2DField) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  UnstructuredIIRBuilder b;
  auto in = b.field("in", LocType::Cells, false);
  auto out = b.field("out", LocType::Cells, false);

  try {
    auto stencil = b.build(
        "OffsetReadsIn2DField",
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                               b.stmt(b.assignExpr(
                                   b.at(out), b.at(in, AccessType::r,
                                                   ast::Offsets{ast::unstructured, false, 1}))))))));
    FAIL() << "Semantic error not thrown";
  } catch(SemanticError& error) {
    SUCCEED();
  }
}

} // anonymous namespace
