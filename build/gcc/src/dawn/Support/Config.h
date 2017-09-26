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
#define DAWN_ON_UNIX 1

// Define if this is an Apple platform 
/* #undef DAWN_ON_APPLE */

// Define if this is a Linux platform 
#define DAWN_ON_LINUX 1

// Major version of DAWN 
#define DAWN_VERSION_MAJOR 0

// Minor version of DAWN 
#define DAWN_VERSION_MINOR 0

// Patch version of DAWN 
#define DAWN_VERSION_PATCH 1

// DAWN version string 
#define DAWN_VERSION_STRING "0.0.1-dev"

#endif
