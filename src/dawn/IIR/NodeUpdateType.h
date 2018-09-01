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

#ifndef DAWN_IIR_NODEUPDATETYPE_H
#define DAWN_IIR_NODEUPDATETYPE_H

namespace dawn {
namespace iir {

enum class NodeUpdateType : int {
  treeAbove = 2,
  levelAndTreeAbove = 1,
  level = 0,
  levelAndTreeBelow = -1,
  treeBelow = -2
};

namespace impl {
bool updateLevel(NodeUpdateType updateType);
bool updateTreeAbove(NodeUpdateType updateType);
bool updateTreeBelow(NodeUpdateType updateType);
} // namespace impl
} // namespace iir
} // namespace dawn

#endif
