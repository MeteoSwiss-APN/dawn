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

#ifndef DAWN_UNITTEST_CODEDUMPER_H
#define DAWN_UNITTEST_CODEDUMPER_H

#include <iostream>

#include "dawn/CodeGen/CXXNaive/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/Cuda/CudaCodeGen.h"
#include "dawn/CodeGen/CodeGen.h"

namespace dawn {

/// @brief Code generation dumper for unit tests
/// @ingroup unittest
class CodeDumper {
public:
  static void dumpNaive(std::ostream& os, dawn::codegen::stencilInstantiationContext& ctx);
  static void dumpCuda(std::ostream& os, dawn::codegen::stencilInstantiationContext& ctx);
};

} // namespace dawn

#endif
