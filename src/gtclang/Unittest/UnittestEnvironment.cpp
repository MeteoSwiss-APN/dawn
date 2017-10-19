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

#include "dawn/Support/Assert.h"
#include "gtclang/Unittest/UnittestEnvironment.h"

namespace gtclang {

UnittestEnvironment::~UnittestEnvironment() {}

void UnittestEnvironment::SetUp() {}

void UnittestEnvironment::TearDown() {}

std::string UnittestEnvironment::testCaseName() const {
  const ::testing::TestInfo* testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
  DAWN_ASSERT_MSG(testInfo, "testCaseName() called outside a test case");
  return testInfo->test_case_name();
}

std::string UnittestEnvironment::testName() const {
  const ::testing::TestInfo* testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
  DAWN_ASSERT_MSG(testInfo, "testName() called outside a test");
  return testInfo->test_case_name();
}

UnittestEnvironment* UnittestEnvironment::instance_ = nullptr;

UnittestEnvironment& UnittestEnvironment::getSingleton() {
  if(instance_ == nullptr)
    instance_ = new UnittestEnvironment;
  return *instance_;
}

} // namespace gtclang
