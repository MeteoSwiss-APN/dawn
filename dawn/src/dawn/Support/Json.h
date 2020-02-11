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
//
// This file includes the headers of the json library.
// See: https://github.com/nlohmann/json/tree/master
//
//===------------------------------------------------------------------------------------------===//

#ifndef DAWN_SUPPORT_JSON_H
#define DAWN_SUPPORT_JSON_H

#include <nlohmann_json.hpp>

namespace dawn {

namespace json {

/// @class json
/// @brief JSON object
///
/// @see https://github.com/nlohmann/json/tree/master
/// @ingroup support
using nlohmann::json;

} // namespace json

} // namespace dawn

#endif
