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

#ifndef DAWN_SUPPORT_CONFIG_H
#define DAWN_SUPPORT_CONFIG_H

// Define if this is Unixish platform 
#cmakedefine DAWN_ON_UNIX ${DAWN_ON_UNIX}

// Define if this is an Apple platform 
#cmakedefine DAWN_ON_APPLE ${DAWN_ON_APPLE}

// Define if this is a Linux platform 
#cmakedefine DAWN_ON_LINUX ${DAWN_ON_LINUX}

// Major version of DAWN 
#define DAWN_VERSION_MAJOR ${DAWN_VERSION_MAJOR}

// Minor version of DAWN 
#define DAWN_VERSION_MINOR ${DAWN_VERSION_MINOR}

// Patch version of DAWN 
#define DAWN_VERSION_PATCH ${DAWN_VERSION_PATCH}

// DAWN version string 
#define DAWN_VERSION_STRING "${DAWN_VERSION_STRING}"

#endif
