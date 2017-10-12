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

#include "gtclang/Unittest/UnittestEnvironment.h"

namespace gtclang {

UnittestEnvironment::~UnittestEnvironment() {}

void UnittestEnvironment::SetUp() {
    fileManager_.setKind(FileManager::testKind::unittest);
}

void UnittestEnvironment::TearDown() {}

UnittestEnvironment* UnittestEnvironment::instance_ = nullptr;

UnittestEnvironment& UnittestEnvironment::getSingleton() {
  if(instance_ == nullptr) {
    instance_ = new UnittestEnvironment;
  }
  return *instance_;
}

} // namespace gtclang
