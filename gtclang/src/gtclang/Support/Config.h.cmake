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

#ifndef GTCLANG_SUPPORT_CONFIG_H
#define GTCLANG_SUPPORT_CONFIG_H

// Major version of GTClang
#define GTCLANG_VERSION_MAJOR ${VERSION_MAJOR}

// Minor version of GTClang
#define GTCLANG_VERSION_MINOR ${VERSION_MINOR}

// Patch version of GTClang
#define GTCLANG_VERSION_PATCH ${VERSION_PATCH}

// GTClang version string
#define GTCLANG_VERSION_STR "${GTClang_VERSION}"

// GTClang full version string
#define GTCLANG_FULL_VERSION_STR "${GTCLANG_FULL_VERSION}"

// Path to gridtools clang DSL headers
#define GTCLANG_DSL_INCLUDES "${PROJECT_SOURCE_DIR}/src;${CMAKE_INSTALL_FULL_INCLUDEDIR};${DAWN_DRIVER_INCLUDEDIR}"

// Ressource path of the Clang specific headers needed during invocation of Clang
#define GTCLANG_CLANG_RESSOURCE_INCLUDE_PATH "${CLANG_RESSOURCE_INCLUDE_PATH}"

#endif
