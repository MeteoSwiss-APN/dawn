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
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassSetCaches.h"
#include "dawn/Optimizer/PassStageSplitter.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Support/Iterator.h"

#include <fstream>
#include <gtest/gtest.h>

namespace {

using namespace dawn;
using namespace dawn::iir;

class TestPassSetCaches : public ::testing::Test {
protected:
  OptimizerContext::OptimizerContextOptions options_;
  std::unique_ptr<dawn::OptimizerContext> context_;

  explicit TestPassSetCaches() {
    context_ = std::make_unique<OptimizerContext>(options_,
                                                  std::make_shared<SIR>(ast::GridType::Cartesian));
    UIDGenerator::getInstance()->reset();
  }

  void runTest(const std::string& filename, int nMultiStages,
               const std::vector<std::vector<std::string>>& fieldNames,
               const std::vector<std::vector<Cache::CacheType>>& cacheTypes,
               const std::vector<std::vector<Cache::IOPolicy>>& ioPolicies) {
    auto instantiation = IIRSerializer::deserialize(filename);

    // Run stage splitter pass
    PassStageSplitter stageSplitPass(*context_);
    EXPECT_TRUE(stageSplitPass.run(instantiation));

    // Expect pass to succeed...
    PassSetCaches setCachesPass(*context_);
    EXPECT_TRUE(setCachesPass.run(instantiation));

    auto& stencils = instantiation->getStencils();
    ASSERT_GT(stencils.size(), 0);

    auto& stencil = stencils[0];
    auto& multiStages = stencil->getChildren();
    ASSERT_EQ(multiStages.size(), nMultiStages);

    for(auto [i, multiStage] : enumerate(multiStages)) {
      for(auto [j, fieldName] : enumerate(fieldNames[i])) {
        int accessID = stencil->getMetadata().getAccessIDFromName(fieldName);
        ASSERT_TRUE(multiStage->isCached(accessID));
        ASSERT_TRUE(multiStage->getCache(accessID).getType() == cacheTypes[i][j]);
        ASSERT_TRUE(multiStage->getCache(accessID).getIOPolicy() == ioPolicies[i][j]);
      }
    }
  }
};

TEST_F(TestPassSetCaches, IJCacheTest1) {
  /*
    vertical_region(k_start, k_end) {
      tmp = in;
      out = tmp(i + 1);
    } */
  runTest("input/IJCacheTest01.iir", 1, {{"tmp"}}, {{Cache::CacheType::IJ}},
          {{Cache::IOPolicy::local}});
}

TEST_F(TestPassSetCaches, IJCacheTest2) {
  /*
    vertical_region(k_start, k_end) {
      tmp = in;
    }
    vertical_region(k_start, k_end) {
      out = tmp(i + 1);
    } */
  runTest("input/IJCacheTest02.iir", 1, {{"tmp"}}, {{Cache::CacheType::IJ}},
          {{Cache::IOPolicy::local}});
}

TEST_F(TestPassSetCaches, KCacheTest1) {
  /*
    vertical_region(k_end, k_end) {
      tmp = in;
    }
    vertical_region(k_end - 1, k_start) {
      out = tmp(k + 1);
    } */
  runTest("input/KCacheTest01.iir", 1, {{"tmp"}}, {{Cache::CacheType::K}},
          {{Cache::IOPolicy::fill}});
}

TEST_F(TestPassSetCaches, KCacheTest1b) {
  /*
    vertical_region(k_end, k_end) {
      tmp = in;
    }
    vertical_region(k_end - 1, k_start) {
      tmp = in * 2;
      out = tmp(k + 1);
    } */
  runTest("input/KCacheTest01b.iir", 1, {{"tmp"}}, {{Cache::CacheType::K}},
          {{Cache::IOPolicy::local}});
}

TEST_F(TestPassSetCaches, KCacheTest2) {
  /*
    vertical_region(k_start, k_start) {
      tmp = a;
    }
    vertical_region(k_start + 1, k_end) {
      b = tmp(k - 1);
    }
    vertical_region(k_end, k_end) {
      tmp = (b(k - 1) + b) * tmp;
    }
    vertical_region(k_end - 1, k_start) {
      c = tmp(k + 1);
    } */
  runTest("input/KCacheTest02.iir", 2, {{"tmp"}, {"tmp"}},
          {{Cache::CacheType::K}, {Cache::CacheType::K}},
          {{Cache::IOPolicy::fill_and_flush}, {Cache::IOPolicy::fill}});
}

TEST_F(TestPassSetCaches, KCacheTest2b) {
  /*
    vertical_region(k_start, k_start) {
      tmp = a;
    }
    vertical_region(k_start + 1, k_end) {
      b = tmp(k - 1);
    }
    vertical_region(k_end, k_end) {
      tmp = (b(k - 1) + b) * tmp;
    }
    vertical_region(k_end - 1, k_start) {
      tmp = 2 * b;
      c = tmp(k + 1);
    } */
  runTest("input/KCacheTest02b.iir", 2, {{"tmp"}, {"b", "tmp"}},
          {{Cache::CacheType::K}, {Cache::CacheType::K, Cache::CacheType::K}},
          {{Cache::IOPolicy::fill_and_flush}, {Cache::IOPolicy::fill, Cache::IOPolicy::bpfill}});
}

TEST_F(TestPassSetCaches, KCacheTest3) {
  /*
    vertical_region(k_start, k_end) {
      tmp = in;
      b = a;
      c = b(k + 1);
      c = b(k - 1);
      out = tmp;
    } */
  runTest("input/KCacheTest03.iir", 2, {{"tmp"}, {"b", "tmp"}},
          {{Cache::CacheType::K}, {Cache::CacheType::K, Cache::CacheType::K}},
          {{Cache::IOPolicy::flush}, {Cache::IOPolicy::fill, Cache::IOPolicy::fill}});
}

TEST_F(TestPassSetCaches, KCacheTest4) {
  /*
    vertical_region(k_start, k_end) {
      tmp = in;
      b1 = a1;
      c1 = b1(k + 1);
      c1 = b1(k - 1);
      out = tmp;
      tmp = in;
      b2 = a2;
      c2 = b2(k + 1);
      c2 = b2(k - 1);
      out = tmp;
    } */
  auto kCacheType = Cache::CacheType::K;
  auto fillPolicy = Cache::IOPolicy::fill;
  auto flushPolicy = Cache::IOPolicy::flush;
  auto fillAndFlush = Cache::IOPolicy::fill_and_flush;

  runTest("input/KCacheTest04.iir", 3, {{"tmp"}, {"b1", "tmp"}, {"b2", "tmp"}},
          {{kCacheType}, {kCacheType, kCacheType}, {kCacheType, kCacheType}},
          {{flushPolicy}, {fillPolicy, fillAndFlush}, {fillPolicy, fillPolicy}});
}

TEST_F(TestPassSetCaches, KCacheTest5) {
  /*
    stencil Test1 {
      vertical_region(k_start, k_end) {
        out = in + in(k + 1);
      }
    }
    stencil Test2 {
      vertical_region(k_start, k_end) {
        out += in + in(k + 1);
      }
    } */
  auto kCacheType = Cache::CacheType::K;
  auto fillPolicy = Cache::IOPolicy::fill;
  auto fillAndFlush = Cache::IOPolicy::fill_and_flush;

  runTest("input/KCacheTest05_Test1.iir", 1, {{"in"}}, {{kCacheType}}, {{fillPolicy}});
  runTest("input/KCacheTest05_Test2.iir", 1, {{"in", "out"}}, {{kCacheType, kCacheType}},
          {{fillPolicy, fillAndFlush}});
}

TEST_F(TestPassSetCaches, KCacheTest6) {
  /*
    vertical_region(k_start, k_start) {
      tmp = a;
    }
    vertical_region(k_start + 1, k_end) {
      tmp = a * 2;
      b = tmp(k - 1);
    }
    vertical_region(k_end - 3, k_end - 3) {
      c = tmp[k + 3] + tmp[k + 2] + tmp[k + 1];
    }
    vertical_region(k_end - 4, k_start) {
      tmp = b;
      c = tmp[k + 1];
    } */
  auto kCacheType = Cache::CacheType::K;
  runTest("input/KCacheTest06.iir", 2, {{"tmp"}, {"tmp"}}, {{kCacheType}, {kCacheType}},
          {{Cache::IOPolicy::epflush}, {Cache::IOPolicy::bpfill}});
}

} // anonymous namespace
