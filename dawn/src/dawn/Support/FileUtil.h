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

#ifndef DAWN_SUPPORT_FILEUTIL_H
#define DAWN_SUPPORT_FILEUTIL_H

#include <fstream>
#include "dawn/Support/StringRef.h"

namespace dawn {

/// @brief Extract the filename from `path`
///
/// This will only work on UNIX like platforms.
///
/// @ingroup support
extern StringRef getFilename(StringRef path);

/// @brief Extract the extension from `filename`
/// @ingroup support
extern StringRef getExtension(StringRef filename);

/// @brief Extract the filename without extension from `path`
/// @ingroup support
extern StringRef getFilenameWithoutExtension(StringRef path);

/// @brief Read the contents of a file into a string
/// @ingroup support
std::string readFile(const std::string& file);

} // namespace dawn

#endif
