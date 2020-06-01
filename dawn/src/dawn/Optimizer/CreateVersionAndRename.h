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

#include "dawn/IIR/ASTFwd.h"
#include <memory>

namespace dawn {
namespace iir {
class StencilInstantiation;
class Stencil;
} // namespace iir
enum class RenameDirection {
  Above, ///< Rename all fields above the current statement
  Below  ///< Rename all fields below the current statement
};

/// @brief Add a new version to the field/local variable given by `AccessID`
///
/// This will create a **new** field and trigger a renaming of all the remaining occurences in the
/// AccessID maps either above or below that statement, starting one statement before or after
/// the current statement. Optionally, an `Expr` can be passed which will be renamed as well
/// (usually the left- or right-hand side of an assignment).
/// Consider the following example:
///
/// @code
///   v = 2 * u
///   lap = u(i+1)
///   u = lap(i+1)
/// @endcode
///
/// We may want to rename `u` in the second statement (an all occurences of `u` above) to
/// resolve the race-condition. We expect to end up with:
///
/// @code
///   v = 2 * u_1
///   lap = u_1(i+1)
///   u = lap(i+1)
/// @endcode
///
/// where `u_1` is the newly created version of `u`.
///
/// @param AccessID   AccessID of the field for which a new version will be created
/// @param stencil    Current stencil
/// @param stageIdx   **Linear** index of the stage in the stencil
/// @param stmtIdx    Index of the statement inside the stage
/// @param expr       Expression to be renamed (usually the left- or right-hand side of an
///                   assignment). Can be `NULL`.
/// @returns AccessID of the new field
int createVersionAndRename(iir::StencilInstantiation* instantiation, int AccessID,
                           iir::Stencil* stencil, int curStageIdx, int curStmtIdx,
                           std::shared_ptr<iir::Expr>& expr, RenameDirection dir);

} // namespace dawn
