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

#ifndef DAWN_SUPPORT_SOURCELOCATION_H
#define DAWN_SUPPORT_SOURCELOCATION_H

#include <iosfwd>

namespace dawn {

/// @brief Location (line, col) in a specific file (-1 represents unknown location)
///
/// @ingroup support
struct SourceLocation {
  /// @name Constructors
  /// @{
  SourceLocation(int line, int column) : Line(line), Column(column) {}
  SourceLocation() : Line(-1), Column(-1) {}
  SourceLocation(const SourceLocation&) = default;
  SourceLocation(SourceLocation&&) = default;
  SourceLocation& operator=(const SourceLocation&) = default;
  SourceLocation& operator=(SourceLocation&&) = default;
  /// @}

  bool isValid() const { return (Line != -1 && Column != -1); }

  int Line;
  int Column;
};

extern bool operator==(const SourceLocation& a, const SourceLocation& b);
extern bool operator!=(const SourceLocation& a, const SourceLocation& b);
extern std::ostream& operator<<(std::ostream& os, const SourceLocation& sourceLocation);

} // namespace dawn

#endif
