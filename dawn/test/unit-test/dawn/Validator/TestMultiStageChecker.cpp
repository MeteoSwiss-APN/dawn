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
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Support/Exception.h"
#include "dawn/Validator/MultiStageChecker.h"

#include <fstream>
#include <gtest/gtest.h>

using namespace dawn;

namespace {

TEST(TestMultiStageChecker, LaplacianTwoStep) {
  // Load IIR from file
  auto instantiation = IIRSerializer::deserialize("input/LaplacianTwoStep.iir");
  MultiStageChecker checker(instantiation.get(), 0);

  // Run multistage checker and succeed if exception is thrown
  try {
    checker.run();
    FAIL() << "Max halo error not thrown";
  } catch(CompileError& error) {
    SUCCEED();
  }
}

} // anonymous namespace
