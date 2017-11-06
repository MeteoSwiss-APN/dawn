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

#include "dawn/SIR/SIR.h"
#include "dawn/SIR/SIR.pb.h"
#include "dawn/SIR/SIRSerializer.h"
#include "dawn/Support/Format.h"
#include <fstream>
#include <google/protobuf/util/json_util.h>

namespace dawn {

namespace {

std::string serializeImpl(const SIR* sir) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  // Convert SIR to protobuf SIR
  sir::proto::SIR sirProto;

  // 
  
  // SIR
  sirProto.set_filename(sir->Filename);
  
  // Envode message to JSON formatted string
  std::string str;
  auto status = google::protobuf::util::MessageToJsonString(sirProto, &str);
  if(!status.ok())
    throw std::runtime_error(dawn::format("cannot serialize SIR: %s", status.ToString()));
  return str;
}

std::shared_ptr<SIR> deserializeImpl(const std::string& str) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  // Decode JSON formatted string
  sir::proto::SIR sirProto;
  auto status = google::protobuf::util::JsonStringToMessage(str, &sirProto);
  if(!status.ok())
    throw std::runtime_error(dawn::format("cannot deserialize SIR: %s", status.ToString()));

  // Convert protobuf SIR to SIR
  auto sir = std::make_shared<SIR>();
  
  
  
  return sir;
}

} // namespace internal

std::shared_ptr<SIR> SIRSerializer::deserialize(const std::string& file) {
  std::ifstream ifs(file);
  if(!ifs.is_open())
    throw std::runtime_error(
        dawn::format("cannot deserialize SIR: failed to open file \"%s\"", file));

  std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  return deserializeImpl(str);
}

std::shared_ptr<SIR> SIRSerializer::deserializeFromString(const std::string& str) {
  return deserializeImpl(str);
}

void SIRSerializer::serialize(const std::string& file, const SIR* sir) {
  std::ofstream ofs(file);
  if(!ofs.is_open())
    throw std::runtime_error(
        dawn::format("cannot serialize SIR: failed to open file \"%s\"", file));

  auto str = serializeImpl(sir);
  std::copy(str.begin(), str.end(), std::ostreambuf_iterator<char>(ofs));
}

std::string SIRSerializer::serializeToString(const SIR* sir) { return serializeImpl(sir); }

} // namespace dawn
