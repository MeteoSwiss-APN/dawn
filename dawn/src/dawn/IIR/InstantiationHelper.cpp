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
#include "dawn/Support/StringRef.h"

namespace dawn {
namespace iir {

static std::string makeNameImpl(const char* prefix, const std::string& name, int AccessID) {
  return prefix + name + "_" + std::to_string(AccessID);
}

static std::string extractNameImpl(StringRef prefix, const std::string& name) {
  StringRef nameRef(name);

  // Remove leading `prefix`
  std::size_t leadingLocalPos = nameRef.find(prefix);
  nameRef = nameRef.drop_front(leadingLocalPos != StringRef::npos ? prefix.size() : 0);

  // Remove trailing `_X` where X is the AccessID
  std::size_t trailingAccessIDPos = nameRef.find_last_of('_');
  nameRef = nameRef.drop_back(
      trailingAccessIDPos != StringRef::npos ? nameRef.size() - trailingAccessIDPos : 0);

  return nameRef.empty() ? name : nameRef.str();
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

bool InstantiationHelper::isStencilCallCodeGenName(const std::string& name) {
  return StringRef(name).startswith("__code_gen_");
}

} // namespace iir
} // namespace dawn
