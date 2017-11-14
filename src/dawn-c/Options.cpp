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

#include "dawn-c/Options.h"
#include "dawn-c/util/Allocate.h"
#include "dawn-c/util/OptionsWrapper.h"
#include "dawn/Compiler/Options.h"
#include "dawn/Support/Format.h"

using namespace dawn::util;

dawnOptionsEntry_t* dawnOptionsEntryCreateInteger(int value) {
  return OptionsEntryWrapper::construct(value);
}

dawnOptionsEntry_t* dawnOptionsEntryCreateFloat(double value) {
  return OptionsEntryWrapper::construct(value);
}

dawnOptionsEntry_t* dawnOptionsEntryCreateString(const char* value) {
  return OptionsEntryWrapper::construct(value);
}

void dawnOptionsEntryDestroy(dawnOptionsEntry_t* entry) { OptionsEntryWrapper::destroy(entry); }

dawnOptions_t* dawnOptionsCreate() {
  auto wrapper = new OptionsWrapper;
  auto options = allocate<dawnOptions_t>();
  options->Impl = static_cast<void*>(wrapper);
  options->OwnsData = 1;
  return options;
}

void dawnOptionsDestroy(dawnOptions_t* options) {
  if(options) {
    OptionsWrapper* optionsWrapper = toOptionsWrapper(options);
    if(options->OwnsData)
      delete optionsWrapper;
    std::free(options);
  }
}

int dawnOptionsHas(const dawnOptions_t* options, const char* name) {
  const OptionsWrapper* optionsWrapper = toConstOptionsWrapper(options);
  return optionsWrapper->hasOption(name);
}

dawnOptionsEntry_t* dawnOptionsGet(const dawnOptions_t* options, const char* name) {
  const OptionsWrapper* optionsWrapper = toConstOptionsWrapper(options);
  const dawnOptionsEntry_t* entry = optionsWrapper->getOption(name);
  if(!entry)
    dawnFatalError(dawn::format("option '%s' does not exist", name).c_str());
  return OptionsEntryWrapper::copy(entry);
}

void dawnOptionsSet(dawnOptions_t* options, const char* name, const dawnOptionsEntry_t* value) {
  OptionsWrapper* optionsWrapper = toOptionsWrapper(options);
  optionsWrapper->setOptionEntry(name, value);
}

char* dawnOptionsToString(const dawnOptions_t* options) { return nullptr; }
