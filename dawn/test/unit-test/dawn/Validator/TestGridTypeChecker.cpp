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

#include "dawn/AST/Tags.h"
#include "dawn/IIR/ASTFwd.h"
#include "dawn/Unittest/IIRBuilder.h"
#include "dawn/Validator/GridTypeChecker.h"
#include "dawn/Validator/UnstructuredDimensionChecker.h"
#include <gtest/gtest.h>
#include <memory>

using namespace dawn;

namespace {

TEST(GridTypeCheckerTest, MixedGridTypes) {
  using namespace dawn::iir;
  using LocType = dawn::ast::LocationType;

  std::string stencilName("MixedTypes");

  UnstructuredIIRBuilder b;
  auto cell_fA = b.field("cell_field_a", LocType::Cells);
  auto cell_fB = b.field("cell_field_b", LocType::Cells);

  // lets build a consistent stencil
  auto stencilContext =
      b.build(stencilName.c_str(),
              b.stencil(b.multistage(
                  LoopOrderKind::Parallel,
                  b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
                                     b.stmt(b.assignExpr(b.at(cell_fA), b.at(cell_fB))))))));

  // extract the stencil instantiation
  auto si = stencilContext[stencilName.c_str()];

  // lets manually add a field access of the wrong type to the IIR
  ast::Offsets offsets(ast::cartesian_{}, 0, 0, 0);
  auto fieldAccessExprCartesian = std::make_shared<FieldAccessExpr>("wrongFieldAccess", offsets);
  auto exprStmt = iir::makeExprStmt(fieldAccessExprCartesian);
  for(auto& doMethodPtr : iterateIIROver<iir::DoMethod>(*si->getIIR())) {
    doMethodPtr->getAST().push_back(exprStmt);
  }

  // lets ensure that the grid type checking now fails
  GridTypeChecker checker;
  bool consistent = checker.checkGridTypeConsistency(*si->getIIR());
  ASSERT_FALSE(consistent);
}
} // namespace