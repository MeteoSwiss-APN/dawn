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

#ifndef DAWN_UNITTEST_COMPILERUTIL_H
#define DAWN_UNITTEST_COMPILERUTIL_H

#include "dawn/CodeGen/CXXNaive/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/CodeGen.h"
#include "dawn/CodeGen/Cuda/CudaCodeGen.h"
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Compiler/Options.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Serialization/SIRSerializer.h"

#include <fstream>
#include <iostream>

namespace dawn {

using stencilInstantiationContext =
    std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>;

/// @brief Compiler utilities for unit tests
/// @ingroup unittest
class CompilerUtil {
public:
  static stencilInstantiationContext compile(const std::shared_ptr<SIR>& sir);
  static stencilInstantiationContext compile(const std::string& sirFile);
  static const std::shared_ptr<iir::StencilInstantiation>
  load(const std::string& iirFilename,
       const dawn::OptimizerContext::OptimizerContextOptions& options,
       std::unique_ptr<OptimizerContext>& context);
  static void dumpNaive(std::ostream& os, dawn::codegen::stencilInstantiationContext& ctx);
  static void dumpCuda(std::ostream& os, dawn::codegen::stencilInstantiationContext& ctx);
};

} // namespace dawn

#endif
