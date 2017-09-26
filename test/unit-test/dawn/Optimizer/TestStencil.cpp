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

#include "dawn/Optimizer/Stencil.h"
#include <gtest/gtest.h>

using namespace dawn;

namespace {

TEST(StencilTest, StagePosition) {
  Stencil::StagePosition pos1(-1, -1);
  Stencil::StagePosition pos2(0, -1);
  Stencil::StagePosition pos3(0, 0);
  Stencil::StagePosition pos4(1, -1);
  Stencil::StagePosition pos5(1, 0);

  EXPECT_TRUE(pos1 == pos1);
  EXPECT_TRUE(pos1 != pos2);
  EXPECT_TRUE(pos1 < pos2);
  EXPECT_TRUE(pos1 < pos2);
  EXPECT_TRUE(pos1 < pos3);
  EXPECT_TRUE(pos4 < pos5);
}

TEST(StencilTest, StatementPosition) {
  Stencil::StagePosition pos1(0, 0);
  Stencil::StagePosition pos2(1, 1);

  Stencil::StatementPosition stmtPos1(pos1, 0, 0);
  Stencil::StatementPosition stmtPos2(pos1, 0, 1);
  Stencil::StatementPosition stmtPos3(pos1, 1, 0);
  Stencil::StatementPosition stmtPos4(pos2, 0, 0);

  EXPECT_TRUE(stmtPos1 == stmtPos1);
  EXPECT_TRUE(stmtPos1 <= stmtPos1);
  EXPECT_TRUE(stmtPos1 < stmtPos2);
  EXPECT_TRUE(stmtPos1 <= stmtPos2);

  EXPECT_FALSE(stmtPos1 <= stmtPos3);
  EXPECT_TRUE(stmtPos1 != stmtPos3);

  EXPECT_TRUE(stmtPos1 < stmtPos4);
  EXPECT_TRUE(stmtPos1 <= stmtPos4);
  EXPECT_TRUE(stmtPos2 < stmtPos4);
  EXPECT_TRUE(stmtPos2 <= stmtPos4);
  EXPECT_TRUE(stmtPos3 < stmtPos4);
  EXPECT_TRUE(stmtPos3 <= stmtPos4);

  EXPECT_TRUE(stmtPos1.inSameDoMethod(stmtPos1));
  EXPECT_TRUE(stmtPos1.inSameDoMethod(stmtPos2));
  EXPECT_FALSE(stmtPos1.inSameDoMethod(stmtPos3));
}

TEST(StencilTest, LifeTime) {
  Stencil::StagePosition pos1(0, 0);
  Stencil::StagePosition pos2(1, 1);

  Stencil::StatementPosition stmtPos1(pos1, 0, 0);
  Stencil::StatementPosition stmtPos2(pos1, 0, 1);
  Stencil::StatementPosition stmtPos3(pos1, 1, 0);
  Stencil::StatementPosition stmtPos4(pos2, 0, 0);

  Stencil::Lifetime lt1(stmtPos1, stmtPos1);
  Stencil::Lifetime lt2(stmtPos1, stmtPos2);
  Stencil::Lifetime lt3(stmtPos3, stmtPos3);
  Stencil::Lifetime lt4(stmtPos4, stmtPos4);

  EXPECT_TRUE(lt1.overlaps(lt1));
  EXPECT_TRUE(lt1.overlaps(lt2));

  // Same stage but different Do-Method are treated as overlapping!
  EXPECT_TRUE(lt1.overlaps(lt3));
  EXPECT_TRUE(lt2.overlaps(lt3));

  EXPECT_FALSE(lt1.overlaps(lt4));
  EXPECT_FALSE(lt2.overlaps(lt4));
  EXPECT_FALSE(lt3.overlaps(lt4));
}

} // anonymous namespace
