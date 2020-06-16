#include "UnstructuredStencils.h"
#include "dawn/AST/LocationType.h"
#include "dawn/CodeGen/Driver.h"
#include "dawn/Unittest/IIRBuilder.h"

namespace dawn {

std::shared_ptr<iir::StencilInstantiation> getReductionsStencil() {
  UIDGenerator::getInstance()->reset();

  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  iir::UnstructuredIIRBuilder b;
  auto lhs_f = b.field("lhs_field", LocType::Edges);
  auto rhs_f = b.field("rhs_field", LocType::Edges);
  auto cell_f = b.field("cell_field", LocType::Cells);
  auto node_f = b.field("node_field", LocType::Vertices);

  auto stencilInstantiation = b.build(
      "reductions",
      b.stencil(b.multistage(
          LoopOrderKind::Parallel,
          b.stage(LocType::Cells,
                  b.doMethod(
                      dawn::sir::Interval::Start, dawn::sir::Interval::End,
                      b.stmt(b.assignExpr(
                          b.at(lhs_f),
                          b.binaryExpr(b.binaryExpr(b.at(rhs_f),
                                                    b.reduceOverNeighborExpr(
                                                        iir::Op::plus, b.at(cell_f), b.lit(0.),
                                                        {ast::LocationType::Edges,
                                                         ast::LocationType::Cells}),
                                                    iir::Op::plus),
                                       b.reduceOverNeighborExpr(
                                           iir::Op::plus, b.at(node_f), b.lit(0.),
                                           {ast::LocationType::Edges, ast::LocationType::Vertices}),
                                       iir::Op::plus))))))));
  return stencilInstantiation;
}

void runTest(const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation,
             codegen::Backend backend) {
  dawn::codegen::Options options;

  auto tu = dawn::codegen::run(stencilInstantiation, backend, options);
  const std::string code = dawn::codegen::generate(tu);
}

} // namespace dawn