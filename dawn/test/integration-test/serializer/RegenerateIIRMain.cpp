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
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Support/UIDGenerator.h"
#include <memory>

int main(int argc, char* argv[]) {
  dawn::IIRSerializer::serialize("reference_iir/copy_stencil.iir",
                                 createCopyStencilIIRInMemory(dawn::ast::GridType::Cartesian),
                                 dawn::IIRSerializer::Format::Json);

  dawn::UIDGenerator::getInstance()->reset();
  dawn::IIRSerializer::serialize("reference_iir/lap_stencil.iir",
                                 createLapStencilIIRInMemory(dawn::ast::GridType::Cartesian),
                                 dawn::IIRSerializer::Format::Json);

  dawn::UIDGenerator::getInstance()->reset();
  dawn::IIRSerializer::serialize("reference_iir/unstructured_sum_edge_to_cells.iir",
                                 createUnstructuredSumEdgeToCellsIIRInMemory(),
                                 dawn::IIRSerializer::Format::Json);

  dawn::UIDGenerator::getInstance()->reset();
  dawn::IIRSerializer::serialize("reference_iir/unstructured_mixed_copies.iir",
                                 createUnstructuredMixedCopies(),
                                 dawn::IIRSerializer::Format::Json);
}
