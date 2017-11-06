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

#include "dawn/SIR/SIR.h"
#include "dawn/SIR/SIRSerializer.h"
#include "dawn/Unittest/ASTSimplifier.h"
#include "dawn/Unittest/PrintAllExpressionTypes.h"
#include <gtest/gtest.h>

using namespace dawn;
using namespace astgen;

namespace {

TEST(SIRSerializer, Serialize) {
  SIR sir;
  sir.Filename = "test";
  
  auto stencil = std::make_shared<sir::Stencil>();
  stencil->Name = "foo";
  stencil->Fields.emplace_back(std::make_shared<sir::Field>("foo", SourceLocation(10, 15)));
  
  auto stencilBodyAst = std::make_shared<AST>();
  stencilBodyAst->setRoot(block(binop(lit("5"), "+", var("a")), assign(var("a"), var("b"))));
  
  auto vr = std::make_shared<sir::VerticalRegion>(
      stencilBodyAst, std::make_shared<sir::Interval>(sir::Interval::Start, 12, 1, -3),
      sir::VerticalRegion::LK_Forward);

  auto stencilDescAst = std::make_shared<AST>();
  stencilDescAst->setRoot(block(verticalRegion(vr)));
  
  stencil->StencilDescAst = stencilDescAst;
  sir.Stencils.emplace_back(stencil);
  
  std::cout << SIRSerializer::serializeToString(&sir) << std::endl;
}

TEST(SIRSerializer, Deserialize) {

}

} // anonymous namespace
