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

#include "dawn/CodeGen/Options.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Unittest/IIRBuilder.h"

#include <memory>
#include <string>

namespace dawn {

std::shared_ptr<iir::StencilInstantiation> getReductionsStencil();

void runTest(const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation,
             codegen::Backend backend, const std::string& refFile);

} // namespace dawn