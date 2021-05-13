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

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "dawn/AST/IterationSpace.h"
#include "dawn/AST/LocationType.h"
#include "dawn/CodeGen/Options.h"

namespace dawn {
namespace codegen {
namespace cudaico {

std::string chainToTableString(const ast::UnstructuredIterationSpace iterSpace);

std::string chainToSparseSizeString(const ast::UnstructuredIterationSpace iterSpace);

std::string chainToDenseSizeStringHostMesh(std::vector<dawn::ast::LocationType> locs);

std::string chainToVectorString(std::vector<dawn::ast::LocationType> locs);

std::string locToDenseSizeStringGpuMesh(dawn::ast::LocationType loc);

std::string locToStrideString(dawn::ast::LocationType loc);

std::string locToDenseTypeString(dawn::ast::LocationType loc);

std::string locToSparseTypeString(dawn::ast::LocationType loc);

std::string locToStringPlural(dawn::ast::LocationType loc);

} // namespace cudaico
} // namespace codegen
} // namespace dawn