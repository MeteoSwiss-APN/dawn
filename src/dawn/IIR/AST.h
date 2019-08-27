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

#ifndef DAWN_IIR_AST_H
#define DAWN_IIR_AST_H

#include "dawn/AST/AST.h"

namespace dawn {
namespace iir {
//
// TODO refactor_AST: this is TEMPORARY, should be changed in the future to template specialization
//
using AST = ast::AST;
} // namespace iir
} // namespace dawn

#endif
