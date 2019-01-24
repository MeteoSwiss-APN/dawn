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
#include <string>

namespace dawn {

struct SIR;

/// @brief Serialize/Deserialize SIR
/// @ingroup sir
class SIRSerializer {
public:
  SIRSerializer() = delete;

  /// @brief Type of serialization algorithm to use
  enum SerializationKind {
    SK_Json, ///< JSON serialization
    SK_Byte  ///< Protobuf's internal byte format
  };

  /// @brief Deserialize the SIR from `file`
  ///
  /// @param file   Path the file
  /// @param kind   The kind of serialization used in `file` (Json or Byte)
  /// @throws std::excetpion    Failed to deserialize
  /// @returns newly allocated SIR on success or `NULL`
  static std::shared_ptr<SIR> deserialize(const std::string& file,
                                          SerializationKind kind = SK_Json);

  /// @brief Deserialize the SIR from the given JSON formatted `string`
  ///
  /// @param str    Byte or JSON string to deserializee
  /// @param kind   The kind of serialization used in `str` (Json or Byte)
  /// @throws std::excetpion    Failed to deserialize
  /// @returns newly allocated SIR on success or `NULL`
  static std::shared_ptr<SIR> deserializeFromString(const std::string& str,
                                                    SerializationKind kind = SK_Json);

  /// @brief Serialize the SIR as a Json or Byte formatted string to `file`
  ///
  /// @param file   Path the file
  /// @param sir    SIR to serialize
  /// @param kind   The kind of serialization to use to write to `file` (Json or Byte)
  /// @throws std::excetpion    Failed to open `file`
  static void serialize(const std::string& file, const SIR* sir, SerializationKind kind = SK_Json);

  /// @brief Serialize the SIR as a Json or Byte formatted string
  ///
  /// @param sir    SIR to serialize
  /// @param kind   The kind of serialization to use when writing to the string (Json or Byte)
  /// @returns JSON formatted strong of `sir`
  static std::string serializeToString(const SIR* sir, SerializationKind kind = SK_Json);
};

} // namespace dawn

#endif
