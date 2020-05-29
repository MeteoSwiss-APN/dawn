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

#include <iosfwd>
#include <ostream>

namespace dawn {

/// @brief Location (line, col) in a specific file (-1 represents unknown location)
///
/// @ingroup support
struct SourceLocation {

  enum class ReservedSL { Generated = -2 };

  /// @name Constructors
  /// @{
  SourceLocation(int line, int column) : Line(line), Column(column) {}
  SourceLocation() : Line(-1), Column(-1) {}
  SourceLocation(const SourceLocation&) = default;
  SourceLocation(SourceLocation&&) = default;
  SourceLocation& operator=(const SourceLocation&) = default;
  SourceLocation& operator=(SourceLocation&&) = default;
  /// @}

  bool isValid() const { return (Line >= 0 && Column >= 0); }

  explicit operator std::string() const;

  int Line;
  int Column;
};

extern bool operator==(const SourceLocation& a, const SourceLocation& b);
extern bool operator!=(const SourceLocation& a, const SourceLocation& b);
extern std::ostream& operator<<(std::ostream& os, const SourceLocation& sourceLocation);

} // namespace dawn
