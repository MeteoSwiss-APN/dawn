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

#include "dawn/IIR/Cache.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/Array.h"
#include <string>

namespace dawn {
namespace codegen {
namespace cuda {

class CodeGeneratorHelper {
public:
  static std::string generateStrideName(int dim, Array3i fieldDims);
  static std::string indexIteratorName(Array3i dims);
};

} // namespace cuda
} // namespace codegen
} // namespace dawn

#endif
