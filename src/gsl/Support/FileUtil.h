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

#ifndef GSL_SUPPORT_FILEUTIL_H
#define GSL_SUPPORT_FILEUTIL_H

#include "gsl/Support/StringRef.h"

namespace gsl {

/// @brief Extract the filename from `path`
///
/// This will only work on UNIX like platforms.
///
/// @ingroup support
extern StringRef getFilename(StringRef path);

/// @brief Extract the filename without extension from `path`
/// @ingroup support
extern StringRef getFilenameWithoutExtension(StringRef path);

} // namespace gsl

#endif
