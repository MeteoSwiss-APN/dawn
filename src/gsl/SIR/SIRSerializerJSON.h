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

#ifndef GSL_SIR_SIRSERIALIZERJSON_H
#define GSL_SIR_SIRSERIALIZERJSON_H

#include <memory>

namespace gsl {

struct SIR;

/// @brief Serialize/Deserialize SIR
/// @ingroup sir
class SIRSerializerJSON {
public:
  /// @brief Deserialize the SIR from `file`
  ///
  /// @throws std::excetpion    Failed to deserialize
  /// @returns newly allocated SIR on success or `NULL`
  static std::shared_ptr<SIR> deserialize(const std::string& file);

  /// @brief Serialize the SIR to `file`
  ///
  /// @param file   File to serialize
  /// @param sir    SIR to serialize
  /// @throws std::excetpion    Failed to open `file`
  static void serialize(const std::string& file, const SIR* sir);
};

} // namespace gsl

#endif
