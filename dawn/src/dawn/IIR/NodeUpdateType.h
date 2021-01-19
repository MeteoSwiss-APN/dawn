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

namespace dawn {
namespace iir {

enum class NodeUpdateType : int {
  treeAbove = 2,          // update only the tree above the node
  levelAndTreeAbove = 1,  // update current level and tree above the node
  level = 0,              // update current level
  levelAndTreeBelow = -1, // update current level and the tree below
  treeBelow = -2          // update only the tree below the node
};

namespace impl {
/// @brief return true if the current level needs to be updated
bool updateLevel(NodeUpdateType updateType);
/// @brief return true if the tree above the current level needs to be updated
bool updateTreeAbove(NodeUpdateType updateType);
/// @brief return true if the tree below the current level needs to be updated
bool updateTreeBelow(NodeUpdateType updateType);
} // namespace impl
} // namespace iir
} // namespace dawn
