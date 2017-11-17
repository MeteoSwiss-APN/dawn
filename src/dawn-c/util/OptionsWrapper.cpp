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

Options OptionsWrapper::toOptions() const noexcept {
  Options opt;
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  opt.NAME = OptionsEntryWrapper::getValue<TYPE>(options_.find(#NAME)->second);
#include "dawn/Compiler/Options.inc"
#undef OPT
  return opt;
}

char* OptionsWrapper::toString() const { return nullptr; }

} // namespace util

} // namespace dawn
