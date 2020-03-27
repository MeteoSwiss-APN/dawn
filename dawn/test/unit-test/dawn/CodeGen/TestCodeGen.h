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

//#include "dawn/Optimizer/OptimizerContext.h"
//#include "dawn/Optimizer/PassComputeStageExtents.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/FileUtil.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "dawn/Unittest/IIRBuilder.h"

#include <fstream>
#include <gtest/gtest.h>

namespace {

using namespace dawn;
using namespace dawn::iir;
using SInterval = dawn::sir::Interval;

class TestCodeGen : public ::testing::Test {
protected:
  std::shared_ptr<StencilInstantiation> getGlobalIndexStencil() {
    UIDGenerator::getInstance()->reset();

    CartesianIIRBuilder b;
    auto in_f = b.field("in_field", FieldType::ijk);
    auto out_f = b.field("out_field", FieldType::ijk);

    auto stencil_inst =
        b.build("generated",
                b.stencil(b.multistage(
                    LoopOrderKind::Parallel,
                    b.stage(b.doMethod(SInterval::Start, SInterval::End,
                                       b.block(b.stmt(b.assignExpr(b.at(out_f), b.at(in_f)))))),
                    b.stage(1, {0, 2},
                            b.doMethod(SInterval::Start, SInterval::End,
                                       b.block(b.stmt(b.assignExpr(b.at(out_f), b.lit(10)))))))));

    return stencil_inst;
  }

  std::shared_ptr<StencilInstantiation> getLaplacianStencil() {
    UIDGenerator::getInstance()->reset();

    CartesianIIRBuilder b;
    auto in = b.field("in", FieldType::ijk);
    auto out = b.field("out", FieldType::ijk);
    auto dx = b.localvar("dx", BuiltinTypeID::Double);

    auto stencil_inst = b.build(
        "generated",
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
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
                            Op::multiply),
                        b.binaryExpr(b.at(dx), b.at(dx), Op::multiply), Op::divide)))))))));

    return stencil_inst;
  }

  std::shared_ptr<StencilInstantiation> getNonOverlappingInterval() {
    UIDGenerator::getInstance()->reset();

    CartesianIIRBuilder b;
    auto in = b.field("in", FieldType::ijk);
    auto out = b.field("out", FieldType::ijk);
    auto dx = b.localvar("dx", BuiltinTypeID::Double);

    auto stencil_inst = b.build(
        "generated",
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
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
                            Op::multiply),
                        b.binaryExpr(b.at(dx), b.at(dx), Op::multiply), Op::divide)))))),
            b.stage(b.doMethod(SInterval(15, SInterval::End),
                               b.block(b.stmt(b.assignExpr(b.at(out), b.lit(10)))))))));

    return stencil_inst;
  }

  std::shared_ptr<StencilInstantiation> getStencilFromIIR(const std::string& name) {
    OptimizerContext::OptimizerContextOptions options;
    std::unique_ptr<OptimizerContext> context;
    UIDGenerator::getInstance()->reset();

    return CompilerUtil::load("../input/" + name + ".iir", options, context);
  }

  std::shared_ptr<StencilInstantiation> getConditionalStencil() {
    return getStencilFromIIR("conditional_stencil");
  }

  void runTest(const std::shared_ptr<StencilInstantiation> stencil_inst,
               const std::string& ref_file) {
    std::ostringstream oss;
    if(ref_file.find(".cu") != std::string::npos) {
      CompilerUtil::dumpCuda(oss, stencil_inst);
    } else {
      CompilerUtil::dumpNaive(oss, stencil_inst);
    }

    std::string ref = readFile("../reference/" + ref_file);
    ASSERT_EQ(oss.str(), ref) << "Generated code does not match reference code";
  }
};
} // anonymous namespace
