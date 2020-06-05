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

#include "dawn/Support/StringUtil.h"
#include <sstream>

namespace dawn {

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

std::string indent(const std::string& string, int amount) {
  // This could probably be done faster (it's not really speed-critical though)
  std::istringstream iss(string);
  std::ostringstream oss;
  std::string spacer(amount, ' ');
  bool firstLine = true;
  for(std::string line; std::getline(iss, line);) {
    if(!firstLine)
      oss << spacer;
    oss << line;
    if(!iss.eof())
      oss << "\n";
    firstLine = false;
  }
  return oss.str();
}

bool equalsLower(const std::string& a, const std::string& b) {
  return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                    [](char a, char b) { return tolower(a) == tolower(b); });
}

bool startWithLower(std::string str, std::string match) {
  // Convert str to lower case
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  // Convert match to lower case
  std::transform(match.begin(), match.end(), match.begin(), ::tolower);
  if(str.find(match) == 0)
    return true;
  else
    return false;
}

bool endsWithLower(std::string str, std::string match) {
  // Convert str to lower case
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  // Convert match to lower case
  std::transform(match.begin(), match.end(), match.begin(), ::tolower);
  if(str.find(match) + match.size() == str.size())
    return true;
  else
    return false;
}

} // namespace dawn
