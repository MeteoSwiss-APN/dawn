//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _
//                        | |     | |
//                    __ _| |_ ___| | __ _ _ __   __ _
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gtclang/Unittest/IRSplitter.h"

namespace gtclang {

void IRSplitter::split(const std::string& dslFile) {
  std::vector<std::string> flags = {"-std=c++11"};
  dawn::UIDGenerator::getInstance()->reset();
  std::pair<bool, std::shared_ptr<dawn::SIR>> tuple =
      GTClang::run({dslFile, "-fno-codegen"}, flags);
  if(tuple.first) {
    // We have the SIR!
    std::shared_ptr<dawn::SIR> sir = tuple.second;

    // Serialize it...
    dawn::SIRSerializer::serialize(dslFile + ".sir", sir.get());

    // Now compile to IIR!
  }
}

} // namespace gtclang
