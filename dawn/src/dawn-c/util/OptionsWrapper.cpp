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

#include "dawn-c/util/OptionsWrapper.h"
#include "dawn-c/util/Allocate.h"
#include <sstream>

namespace dawn {

namespace util {

OptionsWrapper::~OptionsWrapper() {
  for(auto& optPair : options_)
    OptionsEntryWrapper::destroy(optPair.second);
}

bool OptionsWrapper::hasOption(std::string name) const noexcept {
  return (options_.find(name) != options_.end());
}

const dawnOptionsEntry_t* OptionsWrapper::getOption(std::string name) const noexcept {
  auto it = options_.find(name);
  return it == options_.end() ? nullptr : it->second;
}

void OptionsWrapper::setDawnOptions(dawn::Options* options) const noexcept {
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  options->NAME = OptionsEntryWrapper::getValue<TYPE>(options_.find(#NAME)->second);
#include "dawn/CodeGen/Options.inc"
#include "dawn/Compiler/Options.inc"
#include "dawn/Optimizer/Options.inc"
#include "dawn/Optimizer/PassOptions.inc"
#undef OPT
}

char* OptionsWrapper::toString() const {
  std::stringstream ss;
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  ss << #NAME " = " << OptionsEntryWrapper::getValue<TYPE>(options_.find(#NAME)->second) << "\n";
#include "dawn/CodeGen/Options.inc"
#include "dawn/Compiler/Options.inc"
#include "dawn/Optimizer/Options.inc"
#include "dawn/Optimizer/PassOptions.inc"
#undef OPT
  return allocateAndCopyString(ss.str());
}

} // namespace util

} // namespace dawn
