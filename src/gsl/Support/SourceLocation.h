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

#ifndef GSL_SUPPORT_SOURCELOCATION_H
#define GSL_SUPPORT_SOURCELOCATION_H

#include <iosfwd>

namespace gsl {

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

} // namespace gsl

#endif
