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

#include "gtclang/Support/StringUtil.h"

#include <cstring>
#include <string>

namespace gtclang {

std::string splitString(const std::string& str, std::size_t lineSize, std::size_t indentSize,
                        bool indentFirstLine) {

  // Tokenize string
  std::vector<std::string> tokens(tokenizeString(str, " "));

  std::string resultString;
  if(indentFirstLine && indentSize > 0)
    resultString = std::string(indentSize, ' ');

  resultString += tokens[0];

  std::size_t curLineSize = resultString.size();
  if(!indentFirstLine)
    curLineSize += indentSize;

  for(std::size_t i = 1; i < tokens.size(); ++i) {

    // If we have '\n' in our token, we need to indent the new line if 'indentSize' is not 0
    std::vector<std::string> curTokens = tokenizeString(tokens[i], "\n");
    for(std::size_t j = 0; j < curTokens.size(); ++j) {
      std::string& curToken = curTokens[j];

      if(!curToken.empty()) {
        curLineSize += curToken.size() + 1;

        // Skip to new line?
        if(curLineSize > lineSize) {
          resultString += '\n';
          curLineSize = curToken.size() + 1;

          // Indent new line?
          if(indentSize > 0) {
            resultString += std::string(indentSize, ' ');
            curLineSize += indentSize;
          }
        } else
          resultString += " ";

        // Append string
        resultString.append(std::move(curToken));
      }

      if(indentSize > 0 && (j != (curTokens.size() - 1))) {
        // the next loop will append a " "
        resultString += "\n" + std::string(indentSize - 1, ' ');
        curLineSize = indentSize;
      }
    }
  }

  return resultString;
}

std::vector<std::string> tokenizeString(const std::string& str, std::string delim) {
  std::vector<std::string> tokensVector;

  std::size_t curPos = 0, actualPos = 0, delimPos = std::string::npos;

  while(true) {
    for(std::size_t i = 0; i < delim.size(); ++i)
      delimPos = std::min(delimPos, str.find_first_of(delim[i], curPos == 0 ? 0 : curPos + 1));

    if(delimPos == 0) {
      curPos = 1;
    } else if(delimPos != std::string::npos) {
      actualPos = curPos == 0 ? 0 : curPos + 1;
      tokensVector.push_back(str.substr(actualPos, delimPos - actualPos));
      curPos = delimPos;
    } else
      break;

    delimPos = std::string::npos;
  }

  tokensVector.push_back(str.substr(curPos <= 1 ? curPos : curPos + 1));
  return tokensVector;
}

const char* copyCString(const std::string& str) {
  const std::string::size_type size = str.size();
  char* buffer = new char[size + 1];
  std::memcpy(buffer, str.c_str(), size + 1);
  return buffer;
}

} // namespace gtclang
