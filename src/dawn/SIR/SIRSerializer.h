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

#ifndef DAWN_SIR_SIRSERIALIZER_H
#define DAWN_SIR_SIRSERIALIZER_H

#include <memory>

namespace dawn {

struct SIR;

/// @brief Serialize/Deserialize SIR
/// @ingroup sir
class SIRSerializer {
public:
  SIRSerializer() = delete;

  /// @brief Deserialize the SIR from `file`
  ///
  /// @throws std::excetpion    Failed to deserialize
  /// @returns newly allocated SIR on success or `NULL`
  static std::shared_ptr<SIR> deserialize(const std::string& file);

  /// @brief Deserialize the SIR from the given JSON formatted `string`
  ///
  /// @throws std::excetpion    Failed to deserialize
  /// @returns newly allocated SIR on success or `NULL`
  static std::shared_ptr<SIR> deserializeFromString(const std::string& str);

  /// @brief Serialize the SIR as a JSON formatted string to `file`
  ///
  /// @throws std::excetpion    Failed to open `file`
  static void serialize(const std::string& file, const SIR* sir);

  /// @brief Serialize the SIR to a JSON string
  /// @returns JSON formatted strong of `sir`
  static std::string serializeToString(const SIR* sir);
};

} // namespace dawn

#endif
