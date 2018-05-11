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

#ifndef DAWN_SUPPORT_INDEXGENERATOR_H
#define DAWN_SUPPORT_INDEXGENERATOR_H

#include <memory>
#include <limits>
#include "dawn/Support/Assert.h"

namespace dawn {

class IndexGenerator {
private:
  IndexGenerator(const IndexGenerator&) = delete;
  IndexGenerator& operator=(const IndexGenerator&) = delete;

  static std::unique_ptr<IndexGenerator> instance;

  long unsigned int idx_ = 0;

private:
  IndexGenerator() = default;

public:
  static IndexGenerator& Instance() {
    if(!instance)
      instance.reset(new IndexGenerator);

    return *(instance.get());
  }

  long unsigned int getIndex() {
    DAWN_ASSERT(idx_ < std::numeric_limits<long unsigned int>::max());
    return idx_++;
  }
};

} // namespace dawn
#endif
