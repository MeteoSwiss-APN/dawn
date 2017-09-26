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

#ifndef DAWN_SIR_SIRSERIALIZERJSON_H
#define DAWN_SIR_SIRSERIALIZERJSON_H

#include <memory>

namespace dawn {

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

} // namespace dawn

#endif
