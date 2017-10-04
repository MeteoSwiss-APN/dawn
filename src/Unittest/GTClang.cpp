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

#include "gtclang/Unittest/GTClang.h"
#include "gtclang/Driver/Driver.h"
#include "gtclang/Support/StringUtil.h"
#include "gtclang/Unittest/Config.h"

namespace gtclang {

bool GTClang::run(const std::vector<std::string>& gtclangFlags,
                  const std::vector<std::string>& clangFlags) {
  llvm::SmallVector<const char*, 16> args;
  args.push_back(GTCLANG_EXECUTABLE);

  for(const auto& flag : gtclangFlags)
    args.push_back(copyCString(flag));

  for(const auto& flag : clangFlags)
    args.push_back(copyCString(flag));

  bool ret = (Driver::run(args) == 0);

  for(std::size_t i = 1; i < args.size(); ++i)
    delete args[i];
  return ret;
}

} // namespace gtclang
