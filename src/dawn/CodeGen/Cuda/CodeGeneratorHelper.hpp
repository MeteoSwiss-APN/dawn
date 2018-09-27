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

#ifndef DAWN_CODEGEN_CUDA_CODEGENERATORHELPER_H
#define DAWN_CODEGEN_CUDA_CODEGENERATORHELPER_H

#include <string>
#include "dawn/Support/Array.h"
#include "dawn/IIR/Cache.h"
#include "dawn/IIR/StencilInstantiation.h"

namespace dawn {
namespace codegen {
namespace cuda {

class CodeGeneratorHelper {
public:
  static std::string generateStrideName(int dim, Array3i fieldDims);
};

} // namespace cuda
} // namespace codegen
} // namespace dawn

#endif
