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

#ifndef DAWN_IIR_IIRSERIALIZER_H
#define DAWN_IIR_IIRSERIALIZER_H

#include "dawn/IIR/IIR.h"
#include "dawn/IIR/IIR.pb.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include <memory>
#include <string>

namespace dawn {

struct SIR;

/// @brief Serialize/Deserialize SIR
/// @ingroup sir
class IIRSerializer {
public:
  IIRSerializer() = delete;

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
  static std::shared_ptr<iir::StencilInstantiation> deserialize(const std::string& file,
                          dawn::OptimizerContext* context,
                          SerializationKind kind = SK_Json);

  /// @brief Deserialize the SIR from the given JSON formatted `string`
  ///
  /// @param str    Byte or JSON string to deserializee
  /// @param kind   The kind of serialization used in `str` (Json or Byte)
  /// @throws std::excetpion    Failed to deserialize
  /// @returns newly allocated SIR on success or `NULL`
  static std::shared_ptr<iir::StencilInstantiation> deserializeFromString(const std::string& str,
                                    dawn::OptimizerContext* context,
                                    SerializationKind kind = SK_Json);

  /// @brief Serialize the SIR as a Json or Byte formatted string to `file`
  ///
  /// @param file   Path the file
  /// @param sir    SIR to serialize
  /// @param kind   The kind of serialization to use to write to `file` (Json or Byte)
  /// @throws std::excetpion    Failed to open `file`
  static void serialize(const std::string& file,
                        const std::shared_ptr<iir::StencilInstantiation> instantiation,
                        dawn::IIRSerializer::SerializationKind kind);

  /// @brief Serialize the SIR as a Json or Byte formatted string
  ///
  /// @param sir    SIR to serialize
  /// @param kind   The kind of serialization to use when writing to the string (Json or Byte)
  /// @returns JSON formatted strong of `sir`
  static std::string
  serializeToString(const std::shared_ptr<iir::StencilInstantiation> instantiation,
                    SerializationKind kind = SK_Json);

private:
  /// \brief deserializeImpl
  /// \param str
  /// \param kind
  /// \param target
  static void deserializeImpl(const std::string& str, IIRSerializer::SerializationKind kind,
                       std::shared_ptr<iir::StencilInstantiation>& target);

  /// \brief deserializeIIR
  /// \param target
  /// \param protoIIR
  static void deserializeIIR(std::shared_ptr<iir::StencilInstantiation>& target,
                      const proto::iir::IIR& protoIIR);

  /// \brief deserializeMetaData
  /// \param target
  /// \param protoMetaData
  static void deserializeMetaData(std::shared_ptr<iir::StencilInstantiation>& target,
                           const proto::iir::StencilMetaInfo& protoMetaData);
  /// \brief serializeImpl
  /// \param instantiation
  /// \param kind
  /// \return
  static std::string serializeImpl(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                            dawn::IIRSerializer::SerializationKind kind);
  /// \brief serializeIIR
  /// \param target
  /// \param iir
  static void serializeIIR(proto::iir::StencilInstantiation& target, const std::unique_ptr<iir::IIR>& iir);
  ///
  /// \brief serializeMetaData
  /// \param target
  /// \param metaData
  static void serializeMetaData(proto::iir::StencilInstantiation& target,
                         iir::StencilMetaInformation& metaData);
};

} // namespace dawn

#endif // DAWN_IIR_IIRSERIALIZER_H
