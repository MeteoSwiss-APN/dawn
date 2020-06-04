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

#include "dawn/Optimizer/Options.h"
#include "dawn/Optimizer/PassManager.h"
#include "dawn/Support/NonCopyable.h"
#include <map>
#include <memory>

namespace dawn {

struct SIR;
namespace sir {
struct Stencil;
}
namespace iir {
class StencilInstantiation;
}

// TODO Remove this
/// @brief Update nodes and fill derived info.
void restoreIIR(std::shared_ptr<iir::StencilInstantiation> stencilInstantiation);

/// @brief Naively lower an SIR to a stencil instantiation map
///
/// This only transforms the SIR to IIR and creates some metadata. It will still need to be run with
/// the parallel pass group to create valid IIR.
std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>
toStencilInstantiationMap(const SIR& stencilIR, const Options& options = {});

} // namespace dawn
