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

#ifndef GSL_SUPPORT_PRINTING_H
#define GSL_SUPPORT_PRINTING_H

/// @macro GSL_PRINT_INDENT
/// @brief Indentation used for each level
/// @see gsl::MakeIndent
/// @ingroup support
#define GSL_PRINT_INDENT 2

/// @class MakeIndent
/// @brief Create a c-string of whitespace for a given level of indentation of size
/// (`Level * GSL_PRINT_INDENT`)
///
/// @ingroup support
/// @{
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
/// @}

#endif
