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
#include "IIRPrinter.h"
namespace dawn {
namespace iir {
void IIRPrinter::dumpHeader(const std::string& name) const {
  ss_ << std::string(level_ * 2, ' ') << format("\e[1;3%im", level_) << name << "\n"
      << std::string(level_ * 2, ' ') << "{\n\e[0m";
}
void IIRPrinter::close() const {
  ss_ << std::string(level_ * 2, ' ') << dawn::format("\e[1;3%im}\n\e[0m", level_);
}
}
}
