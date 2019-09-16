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
#include <dawn-c/Compiler.h>
#include <dawn-c/Options.h>
#include <fstream>
#include <iostream>
#include <sstream>

int main(int argc, char* argv[]) {
  if(argc != 2) {
    std::cerr << "Usage: iir-emitter <file>" << std::endl;
    return 1;
  }

  std::ifstream inputFile(argv[1]);
  if(!inputFile.is_open())
    return 1;

  std::stringstream ss;
  ss << inputFile.rdbuf();

  auto options = dawnOptionsCreate();
  auto entry = dawnOptionsEntryCreateInteger(1);
  dawnOptionsSet(options, "SerializeIIR", entry);

  auto str = ss.str();
  dawnCompile(str.c_str(), str.length(), options);

  return 0;
}
