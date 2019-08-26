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

#include "dawn-c/Compiler.h"
#include "dawn-c/TranslationUnit.h"
#include "dawn/SIR/SIR.h"
#include "dawn/SIR/SIRSerializer.h"
#include "dawn/Unittest/ASTSimplifier.h"
#include <cstring>
#include <gtest/gtest.h>

namespace {

static void freeCharArray(char** array, int size) {
  for(int i = 0; i < size; ++i)
    std::free(array[i]);
  std::free(array);
}

TEST(CompilerTest, CompileEmptySIR) {
  std::string sir;
  dawnTranslationUnit_t* TU = dawnCompile(sir.data(), sir.size(), nullptr);

  EXPECT_EQ(dawnTranslationUnitGetStencil(TU, "invalid"), nullptr);

  char** ppDefines;
  int size;
  dawnTranslationUnitGetPPDefines(TU, &ppDefines, &size);
  EXPECT_NE(size, 0);
  EXPECT_NE(ppDefines, nullptr);

  freeCharArray(ppDefines, size);
  dawnTranslationUnitDestroy(TU);
}

TEST(CompilerTest, CompileCopyStencil) {
  using namespace dawn::astgen;

  // Build copy stencil
  //
  //  copy {
  //    storage in, out;
  //
  //    vertical_region(start, end) {
  //      out = in;
  //    }
  //  }
  //
  auto sir = std::make_shared<dawn::SIR>();
  auto stencil = std::make_shared<dawn::sir::Stencil>();
  stencil->Name = "copy";
  stencil->Fields.emplace_back(std::make_shared<dawn::sir::Field>("in"));
  stencil->Fields.emplace_back(std::make_shared<dawn::sir::Field>("out"));

  auto ast = std::make_shared<dawn::sir::AST>(block(assign(field("out"), field("in"))));
  auto vr = std::make_shared<dawn::sir::VerticalRegion>(
      ast,
      std::make_shared<dawn::sir::Interval>(dawn::sir::Interval::Start, dawn::sir::Interval::End),
      dawn::sir::VerticalRegion::LK_Forward);
  stencil->StencilDescAst = std::make_shared<dawn::sir::AST>(block(verticalRegion(vr)));
  sir->Stencils.emplace_back(stencil);

  std::string sirStr =
      dawn::SIRSerializer::serializeToString(sir.get(), dawn::SIRSerializer::SK_Byte);
  dawnTranslationUnit_t* TU = dawnCompile(sirStr.data(), sirStr.size(), nullptr);

  char* copyCode = dawnTranslationUnitGetStencil(TU, "copy");
  EXPECT_NE(copyCode, nullptr);

  char** ppDefines;
  int size;
  dawnTranslationUnitGetPPDefines(TU, &ppDefines, &size);
  EXPECT_NE(size, 0);
  EXPECT_NE(ppDefines, nullptr);

  freeCharArray(ppDefines, size);
  std::free(copyCode);
  dawnTranslationUnitDestroy(TU);
}

} // anonymous namespace
