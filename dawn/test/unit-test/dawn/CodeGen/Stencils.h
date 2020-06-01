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

#include <memory>
#include <string>

namespace dawn {

std::shared_ptr<iir::StencilInstantiation> getGlobalIndexStencil();
std::shared_ptr<iir::StencilInstantiation> getLaplacianStencil();
std::shared_ptr<iir::StencilInstantiation> getNonOverlappingInterval();

void runTest(const std::shared_ptr<dawn::iir::StencilInstantiation> stencilInstantiation,
             codegen::Backend backend, const std::string& ref_file, bool withSync = true);

} // namespace dawn
