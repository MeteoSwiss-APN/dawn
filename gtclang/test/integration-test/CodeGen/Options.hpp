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
#ifndef TEST_INTEGRATIONTEST_CODEGEN_OPTIONS_H
#define TEST_INTEGRATIONTEST_CODEGEN_OPTIONS_H

#include <string>

namespace dawn {
/**
* @class Options
* Singleton data container for program options
*/
class Options /* singleton */
{
private:
  Options() {
    for(int i = 0; i < 4; ++i) {
      m_size[i] = 0;
    }
    m_verify = true;
  }
  Options(const Options&) {}
  ~Options() {}

public:
  static Options& getInstance();

  int m_size[4] = {12, 12, 12, 10};
  bool m_verify;
};
}

#endif
