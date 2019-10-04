/*===----------------------------------------------------------------------------------*- C -*-===*\
 *                          _
 *                         | |
 *                       __| | __ ___      ___ ___
 *                      / _` |/ _` \ \ /\ / / '_  |
 *                     | (_| | (_| |\ V  V /| | | |
 *                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
 *
 *
 *  This file is distributed under the MIT License (MIT).
 *  See LICENSE.txt for details.
 *
\*===------------------------------------------------------------------------------------------===*/

#ifndef DAWN_TEST_UNITTEST_DAWN_OPTIMIZER_PASSES_TESTENVIRONMENT_H
#define DAWN_TEST_UNITTEST_DAWN_OPTIMIZER_PASSES_TESTENVIRONMENT_H

#include <gtest/gtest.h>

class TestEnvironment : public ::testing::Environment {

public:
  static std::string path_;
};

#endif
