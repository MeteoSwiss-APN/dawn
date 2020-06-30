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

#include "dawn/AST/LocationType.h"
#include <map>

namespace dawn {
static const std::map<ast::NeighborChain, int> ICOChainSizes = {

    {{ast::LocationType::Cells, ast::LocationType::Edges}, 3},
    {{ast::LocationType::Cells, ast::LocationType::Vertices}, 3},
    {{ast::LocationType::Edges, ast::LocationType::Cells}, 2},
    {{ast::LocationType::Edges, ast::LocationType::Vertices}, 2},
    {{ast::LocationType::Vertices, ast::LocationType::Cells}, 6},
    {{ast::LocationType::Vertices, ast::LocationType::Edges}, 6},
    {{ast::LocationType::Edges, ast::LocationType::Cells, ast::LocationType::Edges}, 4},
    {{ast::LocationType::Edges, ast::LocationType::Cells, ast::LocationType::Vertices}, 4}

};
}