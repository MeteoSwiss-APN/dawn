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

/// @brief Indentation used for each level
/// @see dawn::MakeIndent
/// @ingroup support
#define DAWN_PRINT_INDENT 2

/// @class MakeIndent
/// @brief Create a c-string of whitespace for a given level of indentation of size
/// (`Level * DAWN_PRINT_INDENT`)
///
/// @ingroup support
template <unsigned Level>
struct MakeIndent;

// This could be made slightly more generic but requires some arcane preprocessor tricks :)

template <>
struct MakeIndent<0> {
  static constexpr const char* value = "";
};

template <>
struct MakeIndent<1> {
  static constexpr const char* value = "  ";
};

template <>
struct MakeIndent<2> {
  static constexpr const char* value = "    ";
};

template <>
struct MakeIndent<3> {
  static constexpr const char* value = "      ";
};

template <>
struct MakeIndent<4> {
  static constexpr const char* value = "        ";
};

} // namespace dawn
