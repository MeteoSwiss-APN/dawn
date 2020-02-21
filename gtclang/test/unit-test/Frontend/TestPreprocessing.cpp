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

#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/UIDGenerator.h"
#include "gtclang/Unittest/GTClang.h"
#include "gtclang/Unittest/UnittestEnvironment.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace gtclang;

namespace {

TEST(PreprocessingTest, Include) {
  auto flags = UnittestEnvironment::getSingleton().getFlagManager().getDefaultFlags();

  std::string sirString1, sirString2;
  {
    const std::string filename = "input/test_stencil_w_include.cpp";
    dawn::UIDGenerator::getInstance()->reset();
    auto [passed, sir] = GTClang::run({filename, "-fno-codegen"}, flags);
    ASSERT_TRUE(passed);
    ASSERT_TRUE(!sir->Stencils.empty());
    sir->Filename = "";
    sirString1 = dawn::SIRSerializer::serializeToString(sir.get());
  }

  {
    const std::string filename = "input/test_stencil_no_include.cpp";
    dawn::UIDGenerator::getInstance()->reset();
    auto [passed, sir] = GTClang::run({filename, "-fno-codegen"}, flags);
    ASSERT_TRUE(passed);
    ASSERT_TRUE(!sir->Stencils.empty());
    sir->Filename = "";
    sirString2 = dawn::SIRSerializer::serializeToString(sir.get());
  }

  ASSERT_EQ(sirString1, sirString2);
}

using LevelKind = dawn::sir::Interval::LevelKind;
using Interval = dawn::sir::Interval;

void checkVerticalRegion(const dawn::sir::VerticalRegion& vr, const Interval& VerticalInterval,
                         const std::optional<Interval>& Interval0,
                         const std::optional<Interval>& Interval1) {
  ASSERT_EQ(*vr.VerticalInterval, VerticalInterval);
  ASSERT_EQ(vr.IterationSpace[0], Interval0);
  ASSERT_EQ(vr.IterationSpace[1], Interval1);
}

TEST(PreprocessingTest, IterationSpaceToken) {
  auto flags = UnittestEnvironment::getSingleton().getFlagManager().getDefaultFlags();
  const std::string filename = "input/test_iteration_space.cpp";
  dawn::UIDGenerator::getInstance()->reset();
  auto [passed, sir] = GTClang::run({filename, "-fno-codegen", "-backend=c++-naive"}, flags);
  ASSERT_TRUE(!sir->Stencils.empty());

  const auto& stencil = sir->Stencils[0];
  const auto& statements = stencil->StencilDescAst->getRoot()->getStatements();

  {
    // iteration_space(i_start, i_end - 1)
    dawn::ast::VerticalRegionDeclStmt* vrds =
        dawn::dyn_cast<dawn::sir::VerticalRegionDeclStmt>(statements[0].get());
    ASSERT_NE(vrds, nullptr);
    checkVerticalRegion(*vrds->getVerticalRegion(), Interval(LevelKind::Start, LevelKind::End),
                        std::optional<Interval>(Interval(LevelKind::Start, LevelKind::End, 0, -1)),
                        std::optional<Interval>());
  }
  {
    // iteration_space(j_start, j_end - 2)
    dawn::ast::VerticalRegionDeclStmt* vrds =
        dawn::dyn_cast<dawn::sir::VerticalRegionDeclStmt>(statements[1].get());
    ASSERT_NE(vrds, nullptr);
    checkVerticalRegion(*vrds->getVerticalRegion(), Interval(LevelKind::Start, LevelKind::End),
                        std::optional<Interval>(),
                        std::optional<Interval>(Interval(LevelKind::Start, LevelKind::End, 0, -2)));
  }
  {
    // iteration_space(i_start, i_start + 1, j_start, j_start + 1, k_start,
    // k_end - 1)
    dawn::ast::VerticalRegionDeclStmt* vrds =
        dawn::dyn_cast<dawn::sir::VerticalRegionDeclStmt>(statements[2].get());
    ASSERT_NE(vrds, nullptr);
    checkVerticalRegion(
        *vrds->getVerticalRegion(), Interval(LevelKind::Start, LevelKind::End, 0, -1),
        std::optional<Interval>(Interval(LevelKind::Start, LevelKind::Start, 0, 1)),
        std::optional<Interval>(Interval(LevelKind::Start, LevelKind::Start, 0, 1)));
  }
}

} // anonymous namespace
