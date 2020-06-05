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

#include "dawn/IIR/InstantiationHelper.h"

namespace dawn {
namespace iir {

static std::string makeNameImpl(const char* prefix, const std::string& name, int AccessID) {
  return prefix + name + "_" + std::to_string(AccessID);
}

static std::string extractNameImpl(const std::string& prefix, const std::string& name) {
  std::string nameRef(name);

  // Remove leading `prefix`
  std::size_t leadingLocalPos = nameRef.find(prefix);
  nameRef = nameRef.substr(leadingLocalPos != std::string::npos ? prefix.size() : 0);

  // Remove trailing `_X` where X is the AccessID
  std::size_t trailingAccessIDPos = nameRef.find_last_of('_');
  nameRef = nameRef.substr(
      0, trailingAccessIDPos != std::string::npos ? nameRef.size() - trailingAccessIDPos : 0);

  return nameRef.empty() ? name : nameRef;
}

std::string InstantiationHelper::makeLocalVariablename(const std::string& name, int AccessID) {
  return makeNameImpl("__local_", name, AccessID);
}

std::string InstantiationHelper::makeTemporaryFieldname(const std::string& name, int AccessID) {
  return makeNameImpl("__tmp_", name, AccessID);
}

std::string InstantiationHelper::extractLocalVariablename(const std::string& name) {
  return extractNameImpl("__local_", name);
}

std::string InstantiationHelper::extractTemporaryFieldname(const std::string& name) {
  return extractNameImpl("__tmp_", name);
}

std::string InstantiationHelper::makeStencilCallCodeGenName(int StencilID) {
  return "__code_gen_" + std::to_string(StencilID);
}

namespace {
bool startswith(std::string string, std::string prefix) {
  return string.substr(0, prefix.size()) == prefix;
}
} // namespace

bool InstantiationHelper::isStencilCallCodeGenName(const std::string& name) {
  return startswith(name, "__code_gen_");
}

} // namespace iir
} // namespace dawn
