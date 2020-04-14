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

#include "dawn/Optimizer/Driver.h"

#include <magic_enum.hpp>
#include <stdexcept>
#include <string>

namespace dawn {

std::list<PassGroup> defaultPassGroups() {
  return {PassGroup::SetStageName, PassGroup::StageReordering, PassGroup::StageMerger,
          PassGroup::SetCaches, PassGroup::SetBlockSize};
}

PassGroup parsePassGroup(const std::string& passGroup) {
  auto group = magic_enum::enum_cast<PassGroup>(passGroup);
  if(group.has_value()) {
    return group.value();
  } else {
    throw std::invalid_argument(std::string("Could not parse pass group name: ") + passGroup);
  }
}

std::string parsePassGroup(PassGroup passGroup) {
  std::string name;
  name = magic_enum::enum_name(passGroup);
  return name;
}

} // namespace dawn
