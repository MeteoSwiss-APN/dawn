//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
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

#ifndef GSL_SUPPORT_JSON_H
#define GSL_SUPPORT_JSON_H

#include "gsl/Support/External/json/json.hpp"

namespace gsl {

namespace json {

/// @class json
/// @brief JSON object
///
/// @see https://github.com/nlohmann/json/tree/master
/// @ingroup support
using nlohmann::json;

} // namespace json

} // namespace gsl

#endif
