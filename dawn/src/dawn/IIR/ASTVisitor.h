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

#include "dawn/AST/ASTVisitor.h"

namespace dawn {
namespace iir {
//
// TODO refactor_AST: this is TEMPORARY, will be removed in the future
//
using ASTVisitor = ast::ASTVisitor;
using ASTVisitorNonConst = ast::ASTVisitorNonConst;
using ASTVisitorForwarding = ast::ASTVisitorForwarding;
using ASTVisitorPostOrder = ast::ASTVisitorPostOrder;
using ASTVisitorForwardingNonConst = ast::ASTVisitorForwardingNonConst;
using ASTVisitorDisabled = ast::ASTVisitorDisabled;
} // namespace iir
} // namespace dawn
