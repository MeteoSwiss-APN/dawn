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
#include "dawn-c/Compiler.h"
#include "dawn-c/Options.h"
#include "dawn-c/TranslationUnit.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>

int main(int argc, char* argv[]) {
  if(argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <SIR file>" << std::endl;
    return 1;
  }

  std::ifstream inputFile(argv[1]);
  if(!inputFile.is_open()) {
    std::cerr << "Could not open file: " << argv[1] << std::endl;
    return 1;
  }

  std::stringstream ss;
  ss << inputFile.rdbuf();

  auto options = dawnOptionsCreate();
  auto backend = dawnOptionsEntryCreateString("c++-naive");
  dawnOptionsSet(options, "Backend", backend);

  auto str = ss.str();
  auto translationUnit = dawnCompile(str.c_str(), str.length(), options);

  std::ofstream ofs("laplacian_stencil_from_standalone.cpp");

  char** ppDefines;
  int numPPDefines;
  dawnTranslationUnitGetPPDefines(translationUnit, &ppDefines, &numPPDefines);
  for(int i = 0; i < numPPDefines; i++) {
    ofs << ppDefines[i] << "\n";
  }

  ofs << dawnTranslationUnitGetGlobals(translationUnit);
  ofs << dawnTranslationUnitGetStencil(translationUnit, "laplacian_stencil");

  ofs.close();

  return 0;
}
