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

#ifndef DAWN_OPTIMIZER_DEPENDENCYGRAPHACCESSESFWD_H
#define DAWN_OPTIMIZER_DEPENDENCYGRAPHACCESSESFWD_H

namespace dawn {

template <typename VertexData = void>
class DependencyGraphAccessesT;

using DependencyGraphAccesses = DependencyGraphAccessesT<void>;
}
#endif
