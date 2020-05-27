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

#include <memory>
#include <string>

namespace dawn {

struct SIR;

/// @brief Serialize/Deserialize SIR
/// @ingroup sir
class SIRSerializer {
public:
  SIRSerializer() = delete;

  /// @brief Serialization format to use
  enum class Format {
    Json, ///< JSON serialization
    Byte  ///< Protobuf's internal byte format
  };

  /// @brief Parse a format string to serialization type
  static Format parseFormatString(const std::string& format);

  /// @brief Deserialize the SIR from `file`
  ///
  /// @param file   Path the file
  /// @param kind   The kind of serialization used in `file` (Json or Byte)
  /// @throws std::exception    Failed to deserialize
  /// @returns newly allocated SIR on success or `NULL`
  static std::shared_ptr<SIR> deserialize(const std::string& file, Format kind = Format::Json);

  /// @brief Deserialize the SIR from the given JSON formatted `string`
  ///
  /// @param str    Byte or JSON string to deserializee
  /// @param kind   The kind of serialization used in `str` (Json or Byte)
  /// @throws std::exception    Failed to deserialize
  /// @returns newly allocated SIR on success or `NULL`
  static std::shared_ptr<SIR> deserializeFromString(const std::string& str,
                                                    Format kind = Format::Json);

  /// @brief Serialize the SIR as a Json or Byte formatted string to `file`
  ///
  /// @param file   Path the file
  /// @param sir    SIR to serialize
  /// @param kind   The kind of serialization to use to write to `file` (Json or Byte)
  /// @throws std::exception    Failed to open `file`
  static void serialize(const std::string& file, const SIR* sir, Format kind = Format::Json);

  /// @brief Serialize the SIR as a Json or Byte formatted string
  ///
  /// @param sir    SIR to serialize
  /// @param kind   The kind of serialization to use when writing to the string (Json or Byte)
  /// @returns JSON formatted strong of `sir`
  static std::string serializeToString(const SIR* sir, Format kind = Format::Json);
};

} // namespace dawn
