//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gsl/Support/StringUtil.h"

namespace gsl {

extern std::string decimalToOrdinal(int dec) {
  std::string decimal = std::to_string(dec);
  std::string suffix;

  switch(decimal.back()) {
  case '1':
    suffix = "st";
    break;
  case '2':
    suffix = "nd";
    break;
  case '3':
    suffix = "rd";
    break;
  default:
    suffix = "th";
  }

  if(dec > 10)
    suffix = "th";

  return decimal + suffix;
}

} // namespace gsl
