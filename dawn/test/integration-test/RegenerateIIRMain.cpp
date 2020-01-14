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

#include "GenerateInMemoryStencils.h"
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Serialization/IIRSerializer.h"

#include <memory>

using namespace dawn;

int main(int argc, char* argv[]) {
  Options compileOptions;
  OptimizerContext::OptimizerContextOptions optimizerOptions;
  DawnCompiler compiler(compileOptions);
  OptimizerContext optimizer(compiler.getDiagnostics(), optimizerOptions,
                             std::make_shared<dawn::SIR>(ast::GridType::Cartesian));

  {
    UIDGenerator::getInstance()->reset();
    auto copy_stencil = createCopyStencilIIRInMemory(optimizer);
    IIRSerializer::serialize("reference_iir/copy_stencil.iir", copy_stencil,
                             IIRSerializer::Format::Json);
  }
  {
    UIDGenerator::getInstance()->reset();
    auto lap_stencil = createLapStencilIIRInMemory(optimizer);
    IIRSerializer::serialize("reference_iir/lap_stencil.iir", lap_stencil,
                             IIRSerializer::Format::Json);
  }
}
