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

#ifndef DAWN_IIR_STENCILFUNCTIONS_FUNCTIONHANDELING_H
#define DAWN_IIR_STENCILFUNCTIONS_FUNCTIONHANDELING_H
#include <memory>

namespace dawn {
namespace iir {
class StencilFunctionInstantiation;
class IIR;
namespace StencilFunctionHandeling {

///
/// @brief deregisterStencilFunction
/// @param stencilFun
/// @param iir
///
extern void deregisterStencilFunction(std::shared_ptr<StencilFunctionInstantiation> stencilFun,
                                      IIR* iir);
///
/// @brief registerStencilFunction
/// @param stencilFun
/// @param iir
///
extern void registerStencilFunction(std::shared_ptr<StencilFunctionInstantiation> stencilFun,
                                    IIR* iir);

extern void finalizeStencilFunctionSetup(std::shared_ptr<StencilFunctionInstantiation> stencilFun,
                                         iir::IIR* iir);
} // namespace StencilFunctionHandeling
} // namespace iir
} // namespace dawn

#endif // DAWN_IIR_STENCILFUNCTIONS_FUNCTIONHANDELING_H
