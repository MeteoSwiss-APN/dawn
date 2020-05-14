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

#include "dawn/CodeGen/Driver.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Support/FileSystem.h"
#include "dawn/Unittest/IIRBuilder.h"

#include <fstream>
#include <gtest/gtest.h>
#include <string>

namespace dawn {

using SInterval = dawn::sir::Interval;

std::shared_ptr<iir::StencilInstantiation> getGlobalIndexStencil() {
  UIDGenerator::getInstance()->reset();

  iir::CartesianIIRBuilder b;
  auto in_f = b.field("in_field", iir::FieldType::ijk);
  auto out_f = b.field("out_field", iir::FieldType::ijk);

  auto stencilInstantiation =
      b.build("generated",
              b.stencil(b.multistage(
                  iir::LoopOrderKind::Parallel,
                  b.stage(b.doMethod(SInterval::Start, SInterval::End,
                                     b.block(b.stmt(b.assignExpr(b.at(out_f), b.at(in_f)))))),
                  b.stage(1, {0, 2},
                          b.doMethod(SInterval::Start, SInterval::End,
                                     b.block(b.stmt(b.assignExpr(b.at(out_f), b.lit(10)))))))));

  return stencilInstantiation;
}

std::shared_ptr<iir::StencilInstantiation> getLaplacianStencil() {
  UIDGenerator::getInstance()->reset();

  iir::CartesianIIRBuilder b;
  auto in = b.field("in", iir::FieldType::ijk);
  auto out = b.field("out", iir::FieldType::ijk);
  auto dx = b.localvar("dx", BuiltinTypeID::Double);

  auto stencilInstantiation = b.build(
      "generated",
      b.stencil(b.multistage(
          iir::LoopOrderKind::Parallel,
          b.stage(b.doMethod(
              SInterval::Start, SInterval::End, b.declareVar(dx),
              b.block(b.stmt(b.assignExpr(
                  b.at(out),
                  b.binaryExpr(
                      b.binaryExpr(
                          b.lit(-4),
                          b.binaryExpr(
                              b.at(in),
                              b.binaryExpr(b.at(in, {1, 0, 0}),
                                           b.binaryExpr(b.at(in, {-1, 0, 0}),
                                                        b.binaryExpr(b.at(in, {0, -1, 0}),
                                                                     b.at(in, {0, 1, 0}))))),
                          iir::Op::multiply),
                      b.binaryExpr(b.at(dx), b.at(dx), iir::Op::multiply), iir::Op::divide)))))))));

  return stencilInstantiation;
}

std::shared_ptr<iir::StencilInstantiation> getNonOverlappingInterval() {
  UIDGenerator::getInstance()->reset();

  iir::CartesianIIRBuilder b;
  auto in = b.field("in", iir::FieldType::ijk);
  auto out = b.field("out", iir::FieldType::ijk);
  auto dx = b.localvar("dx", BuiltinTypeID::Double);

  auto stencilInstantiation = b.build(
      "generated",
      b.stencil(b.multistage(
          iir::LoopOrderKind::Parallel,
          b.stage(b.doMethod(
              SInterval(SInterval::Start, 10), b.declareVar(dx),
              b.block(b.stmt(b.assignExpr(
                  b.at(out),
                  b.binaryExpr(
                      b.binaryExpr(
                          b.lit(-4),
                          b.binaryExpr(
                              b.at(in),
                              b.binaryExpr(b.at(in, {1, 0, 0}),
                                           b.binaryExpr(b.at(in, {-1, 0, 0}),
                                                        b.binaryExpr(b.at(in, {0, -1, 0}),
                                                                     b.at(in, {0, 1, 0}))))),
                          iir::Op::multiply),
                      b.binaryExpr(b.at(dx), b.at(dx), iir::Op::multiply), iir::Op::divide)))))),
          b.stage(b.doMethod(SInterval(15, SInterval::End),
                             b.block(b.stmt(b.assignExpr(b.at(out), b.lit(10)))))))));

  return stencilInstantiation;
}

void runTest(const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation,
             codegen::Backend backend, const std::string& refFile, bool withSync) {
  dawn::codegen::Options options;
  options.RunWithSync = withSync;

  auto tu = dawn::codegen::run(stencilInstantiation, backend, options);
  const std::string code = dawn::codegen::generate(tu);

  std::ifstream t(refFile);
  const std::string ref((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
  ASSERT_EQ(code, ref) << "Generated code does not match reference code";
}

} // namespace dawn
