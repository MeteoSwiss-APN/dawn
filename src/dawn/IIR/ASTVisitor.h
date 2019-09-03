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

#ifndef DAWN_IIR_ASTVISITOR_H
#define DAWN_IIR_ASTVISITOR_H

#include "dawn/AST/ASTVisitor.h"
#include "dawn/IIR/ASTData.h"

namespace dawn {
namespace iir {
using ASTVisitor = ast::ASTVisitor<IIRASTData>;
using ASTVisitorNonConst = ast::ASTVisitorNonConst<IIRASTData>;
using ASTVisitorForwarding = ast::ASTVisitorForwarding<IIRASTData>;
using ASTVisitorPostOrder = ast::ASTVisitorPostOrder<IIRASTData>;
using ASTVisitorForwardingNonConst = ast::ASTVisitorForwardingNonConst<IIRASTData>;
using ASTVisitorDisabled = ast::ASTVisitorDisabled<IIRASTData>;
} // namespace iir
} // namespace dawn

#endif
