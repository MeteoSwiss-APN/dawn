//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _
//                        | |     | |
//                    __ _| |_ ___| | __ _ _ __   __ _
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "dawn/CodeGen/CXXNaive-ico/CXXNaiveCodeGen.h"
#include "dawn/Unittest/IIRBuilder.h"

#include <c++/7/optional>
#include <cstring>
#include <fstream>

template <typename CG>
void dump(std::ostream& os, dawn::codegen::stencilInstantiationContext& ctx) {
  dawn::DiagnosticsEngine diagnostics;
  CG generator(ctx, diagnostics, 0);
  auto tu = generator.generateCode();

  std::ostringstream ss;
  for(auto const& macroDefine : tu->getPPDefines())
    ss << macroDefine << "\n";

  ss << tu->getGlobals();
  for(auto const& s : tu->getStencils())
    ss << s.second;
  os << ss.str();
}

int main() {

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::Expr::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in_field", LocType::Cells);
    auto out_f = b.field("out_field", LocType::Cells);

    auto stencil_instantiation = b.build(
        "copyCell", b.stencil(b.multistage(
                        LoopOrderKind::Parallel,
                        b.stage(b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                          b.stmt(b.assignExpr(b.at(out_f), b.at(in_f))))))));

    std::ofstream of("generated/generated_copyCell.hpp");
    dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
    of.close();
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::Expr::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in_field", LocType::Edges);
    auto out_f = b.field("out_field", LocType::Edges);

    auto stencil_instantiation = b.build(
        "copyEdge",
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(LocType::Edges, b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                              b.stmt(b.assignExpr(b.at(out_f), b.at(in_f))))))));

    std::ofstream of("generated/generated_copyEdge.hpp");
    dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
    of.close();
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::Expr::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in_field", LocType::Edges);
    auto out_f = b.field("out_field", LocType::Cells);

    auto stencil_instantiation =
        b.build("accumulateEdgeToCell",
                b.stencil(b.multistage(
                    LoopOrderKind::Parallel,
                    b.stage(b.vregion(
                        dawn::sir::Interval::Start, dawn::sir::Interval::End,
                        b.stmt(b.assignExpr(b.at(out_f),
                                            b.reduceOverNeighborExpr(
                                                Op::plus, b.at(in_f, HOffsetType::withOffset, 0),
                                                b.lit(0.), LocType::Cells, LocType::Edges))))))));

    std::ofstream of("generated/generated_accumulateEdgeToCell.hpp");
    dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
    of.close();
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::Expr::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in_field", LocType::Cells);
    auto out_f = b.field("out_field", LocType::Cells);

    auto stencil_instantiation = b.build(
        "verticalSum",
        b.stencil(b.multistage(
            LoopOrderKind::Parallel,
            b.stage(LocType::Cells,
                    b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End, 1, -1,
                              b.stmt(b.assignExpr(
                                  b.at(out_f), b.binaryExpr(b.at(in_f, HOffsetType::noOffset, +1),
                                                            b.at(in_f, HOffsetType::noOffset, -1),
                                                            Op::plus))))))));

    std::ofstream of("generated/generated_verticalSum.hpp");
    dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
    of.close();
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::Expr::LocationType;

    UnstructuredIIRBuilder b;
    auto in_f = b.field("in_field", LocType::Cells);
    auto out_f = b.field("out_field", LocType::Cells);
    auto cnt = b.localvar("cnt", dawn::BuiltinTypeID::Integer);

    auto stencil_instantiation = b.build(
        "diffusion",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(b.vregion(
                dawn::sir::Interval::Start, dawn::sir::Interval::End, b.declareVar(cnt),
                b.stmt(b.assignExpr(
                    b.at(cnt), b.reduceOverNeighborExpr(Op::plus, b.lit(1), b.lit(0),
                                                        dawn::ast::Expr::LocationType::Cells,
                                                        dawn::ast::Expr::LocationType::Cells))),
                b.stmt(b.assignExpr(
                    b.at(out_f),
                    b.reduceOverNeighborExpr(Op::plus, b.at(in_f, HOffsetType::withOffset, 0),
                                             b.binaryExpr(b.unaryExpr(b.at(cnt), Op::minus),
                                                          b.at(in_f, HOffsetType::withOffset, 0),
                                                          Op::multiply),
                                             dawn::ast::Expr::LocationType::Cells,
                                             dawn::ast::Expr::LocationType::Cells))),
                b.stmt(b.assignExpr(
                    b.at(out_f),
                    b.binaryExpr(b.at(in_f), b.binaryExpr(b.lit(0.1), b.at(out_f), Op::multiply),
                                 Op::plus))))))));

    std::ofstream of("generated/generated_diffusion.hpp");
    dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
    of.close();
  }

  {
    using namespace dawn::iir;
    using LocType = dawn::ast::Expr::LocationType;

    UnstructuredIIRBuilder b;
    auto cell_f = b.field("cell_field", LocType::Cells);
    auto edge_f = b.field("edge_field", LocType::Cells);

    auto stencil_instantiation = b.build(
        "gradient",
        b.stencil(b.multistage(
            dawn::iir::LoopOrderKind::Parallel,
            b.stage(
                LocType::Edges,
                b.vregion(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                          b.stmt(b.assignExpr(
                              b.at(edge_f), b.reduceOverNeighborExpr<float>(
                                                Op::plus, b.at(cell_f, HOffsetType::withOffset, 0),
                                                b.lit(0.), dawn::ast::Expr::LocationType::Edges,
                                                dawn::ast::Expr::LocationType::Cells,
                                                std::vector<float>({1., 1.})))))),
            b.stage(LocType::Cells,
                    b.vregion(
                        dawn::sir::Interval::Start, dawn::sir::Interval::End,
                        b.stmt(b.assignExpr(
                            b.at(cell_f), b.reduceOverNeighborExpr<float>(
                                              Op::plus, b.at(edge_f, HOffsetType::withOffset, 0),
                                              b.lit(0.), dawn::ast::Expr::LocationType::Cells,
                                              dawn::ast::Expr::LocationType::Edges,
                                              std::vector<float>({0.25, 0.25, 0.25, 0.25})))))))));

    std::ofstream of("generated/generated_gradient.hpp");
    dump<dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(of, stencil_instantiation);
    of.close();
  }

  return 0;
}
