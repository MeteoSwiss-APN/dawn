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

#ifndef DAWN_OPTIMIZER_UTILITY_TEMPORARYHANDELING_H
#define DAWN_OPTIMIZER_UTILITY_TEMPORARYHANDELING_H

#include "dawn/IIR/Stencil.h"

namespace dawn {

//////// TODO: Should we promote this to a pass? //////

extern void promoteLocalVariableToTemporaryField(iir::IIR* iir, iir::Stencil* stencil, int AccessID,
                                                 const iir::Stencil::Lifetime& lifetime);

extern void demoteTemporaryFieldToLocalVariable(iir::IIR* iir, iir::Stencil* stencil, int AccessID,
                                                const iir::Stencil::Lifetime& lifetime);

extern void promoteTemporaryFieldToAllocatedField(iir::IIR* iir, int AccessID);

/// @brief Rename all occurences of field `oldAccessID` to `newAccessID`
extern void renameAllOccurrences(iir::IIR* iir, iir::Stencil* stencil, int oldAccessID,
                                 int newAccessID);
}

#endif // DAWN_OPTIMIZER_UTILITY_TEMPORARYHANDELING_H
