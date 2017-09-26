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

#include "gsl/Support/FileUtil.h"

namespace gsl {

StringRef getFilename(StringRef path) { return path.substr(path.find_last_of('/') + 1); }

StringRef getFilenameWithoutExtension(StringRef path) {
  auto filename = getFilename(path);
  return filename.substr(0, filename.find_last_of("."));
}

} // namespace gsl
