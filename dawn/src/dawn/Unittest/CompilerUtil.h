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

#include "dawn/CodeGen/CodeGen.h"
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Compiler/Options.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Serialization/SIRSerializer.h"

#include <fstream>
#include <iostream>
#include <string>

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
  static void dumpNaive(std::ostream& os, std::shared_ptr<iir::StencilInstantiation> si);
  static void dumpNaiveIco(std::ostream& os, std::shared_ptr<iir::StencilInstantiation> si);
  static void dumpCuda(std::ostream& os, std::shared_ptr<iir::StencilInstantiation> si);
};

} // namespace dawn

#endif
