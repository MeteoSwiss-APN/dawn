//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _     _ _              _            _
//                        (_)   | | |            | |          | |
//               __ _ _ __ _  __| | |_ ___   ___ | |___    ___| | __ _ _ __   __ _
//              / _` | '__| |/ _` | __/ _ \ / _ \| / __|  / __| |/ _` | '_ \ / _` |
//             | (_| | |  | | (_| | || (_) | (_) | \__ \ | (__| | (_| | | | | (_| |
//              \__, |_|  |_|\__,_|\__\___/ \___/|_|___/  \___|_|\__,_|_| |_|\__, |
//               __/ |                                                        __/ |
//              |___/                                                        |___/
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gtclang/Unittest/FlagManager.h"
#include "gtclang/Support/Config.h"

namespace gtclang {

FlagManager::FlagManager() : includes_(GTCLANG_UNITTEST_INCLUDES) {}

std::string FlagManager::getInclude() const { return "-I" + includes_; }

std::vector<std::string> FlagManager::getDefaultFlags() const {
  return std::vector<std::string>{"-std=c++11", getInclude()};
}

} // namespace gtclang
