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

#include "gtclang/Unittest/UnittestEnvironment.h"

namespace gtclang {

UnittestEnvironment::~UnittestEnvironment() {}

void UnittestEnvironment::SetUp() {}

void UnittestEnvironment::TearDown() {}

UnittestEnvironment* UnittestEnvironment::instance_ = nullptr;

UnittestEnvironment& UnittestEnvironment::getSingleton() {
  if(instance_ == nullptr) {
    instance_ = new UnittestEnvironment;
  }
  return *instance_;
}

} // namespace gtclang
