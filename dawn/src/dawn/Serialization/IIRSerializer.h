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
#include "dawn/IIR/IIR/IIR.pb.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include <memory>
#include <string>

namespace dawn {

struct SIR;
namespace iir {
class StencilInstantiation;
}

/// @brief Serialize/Deserialize the internal representation of the user stencils
class IIRSerializer {
public:
  IIRSerializer() = delete;

  /// @brief Serialization format to use
  enum class Format {
    Json, ///< JSON serialization
    Byte  ///< Protobuf's internal byte format
  };

  /// @brief Deserialize the StencilInstantiaion from `file`
  ///
  /// @param file    Path the file
  /// @param kind    The kind of serialization used in `file` (Json or Byte)
  /// @throws std::excetpion    Failed to deserialize
  /// @returns newly allocated IIR on success or `NULL`
  static std::shared_ptr<iir::StencilInstantiation> deserialize(const std::string& file,
                                                                Format kind = Format::Json);

  /// @brief Deserialize the StencilInstantiaion from the given JSON formatted `string`
  ///
  /// @param str    Byte or JSON string to deserializee
  /// @param kind   The kind of serialization used in `str` (Json or Byte)
  /// @param context The OptimizerContext in which we register the Instantiation
  /// @throws std::excetpion    Failed to deserialize
  /// @returns newly allocated IIR on success or `NULL`
  static std::shared_ptr<iir::StencilInstantiation>
  deserializeFromString(const std::string& str, dawn::OptimizerContext* context,
                        Format kind = Format::Json);

  /// @brief Serialize the StencilInstantiaion as a Json or Byte formatted string to `file`
  ///
  /// @param file          Path the file
  /// @param instantiation StencilInstantiaion to serialize
  /// @param kind          The kind of serialization to use to write to `file` (Json or Byte)
  /// @throws std::exception    Failed to open `file`
  static void serialize(const std::string& file,
                        const std::shared_ptr<iir::StencilInstantiation> instantiation,
                        dawn::IIRSerializer::Format kind = Format::Json);

  /// @brief Serialize the StencilInstantiaion as a Json or Byte formatted string
  ///
  /// @param instantiation StencilInstantiaion to serialize
  /// @param kind         The kind of serialization to use when writing to the string (Json or Byte)
  /// @returns JSON formatted string of `StencilInstantiaion`
  static std::string
  serializeToString(const std::shared_ptr<iir::StencilInstantiation> instantiation,
                    Format kind = Format::Json);

private:
  /// @brief The implementation of deserialisation used for string and file. This delegates to the
  /// separate implementations of deserializing the IIR and the Metadata
  ///
  /// @param str    the sting to deserialize
  /// @param kind   The kind of serialization used in `str` (Json or Byte)
  /// @param target The newly creadte StencilInstantiation
  static std::shared_ptr<iir::StencilInstantiation> deserializeImpl(const std::string& str,
                                                                    IIRSerializer::Format kind);

  /// @brief deserializeIIR deserializes the IIR tree
  /// @param target     the StencilInstantiation to insert the IIR into
  /// @param protoIIR   the serialized protobuf version of the IIR
  static void deserializeIIR(std::shared_ptr<iir::StencilInstantiation>& target,
                             const proto::iir::IIR& protoIIR);

  /// \brief deserializeMetaData deserializes all the required Metadata
  /// \param target         the StencilInstantiation to insert the metadata into
  /// \param protoMetaData  the serialized protobuf version of the metadata
  static void deserializeMetaData(std::shared_ptr<iir::StencilInstantiation>& target,
                                  const proto::iir::StencilMetaInfo& protoMetaData);

  /// @brief The implementation of serialisation used for string and file. This delegates to the
  /// separate implementations of serializing the IIR and the Metadata
  ///
  /// @param instantiation  The StencilInstantiation to fill
  /// @param kind           The kind of serialization used in the return value (Json or Byte)
  /// @return               The serialized string
  static std::string serializeImpl(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                                   dawn::IIRSerializer::Format kind);
  /// @brief serializeIIR serializes the IIR tree
  /// @param target     The protobuf version of the StencilInstantiation to serilaize the IIR into
  /// @param iir        The IIR to serialize
  static void serializeIIR(proto::iir::StencilInstantiation& target,
                           const std::unique_ptr<iir::IIR>& iir,
                           const std::set<std::string>& usedBc);
  /// @brief serializeMetaData serializes the Metadata
  /// @param target    The protobuf version of the StencilInstantiation to serilaize the metadata to
  /// @param metaData  The Metadata to serialize
  static void serializeMetaData(proto::iir::StencilInstantiation& target,
                                iir::StencilMetaInformation& metaData);
};

} // namespace dawn

#endif // DAWN_IIR_IIRSERIALIZER_H
