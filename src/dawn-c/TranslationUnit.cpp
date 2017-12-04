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

#include "dawn-c/TranslationUnit.h"
#include "dawn-c/util/Allocate.h"
#include "dawn-c/util/TranslationUnitWrapper.h"

using namespace dawn::util;

void dawnTranslationUnitDestroy(dawnTranslationUnit_t* translationUnit) {
  if(translationUnit) {
    dawn::TranslationUnit* TU = toTranslationUnit(translationUnit);
    if(translationUnit->OwnsData)
      delete TU;
    std::free(translationUnit);
  }
}

void dawnTranslationUnitGetPPDefines(const dawnTranslationUnit_t* translationUnit,
                                     char*** ppDefines, int* size) {
  const dawn::TranslationUnit* TU = toConstTranslationUnit(translationUnit);
  const auto& ppVec = TU->getPPDefines();

  char** ppArray = allocate<char*>(ppVec.size());
  for(std::size_t i = 0; i < ppVec.size(); ++i)
    ppArray[i] = allocateAndCopyString(ppVec[i]);

  *size = ppVec.size();
  *ppDefines = ppArray;
}

char* dawnTranslationUnitGetStencil(const dawnTranslationUnit_t* translationUnit,
                                    const char* name) {
  const dawn::TranslationUnit* TU = toConstTranslationUnit(translationUnit);
  auto it = TU->getStencils().find(name);
  return it == TU->getStencils().end() ? nullptr : allocateAndCopyString(it->second);
}

char* dawnTranslationUnitGetGlobals(const dawnTranslationUnit_t* translationUnit) {
  const dawn::TranslationUnit* TU = toConstTranslationUnit(translationUnit);
  return allocateAndCopyString(TU->getGlobals());
}
