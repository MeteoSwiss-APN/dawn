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

#ifndef GSL_SUPPORT_CONFIG_H
#define GSL_SUPPORT_CONFIG_H

// Define if this is Unixish platform 
#cmakedefine GSL_ON_UNIX ${GSL_ON_UNIX}

// Define if this is an Apple platform 
#cmakedefine GSL_ON_APPLE ${GSL_ON_APPLE}

// Define if this is a Linux platform 
#cmakedefine GSL_ON_LINUX ${GSL_ON_LINUX}

// Major version of GSL 
#define GSL_VERSION_MAJOR ${GSL_VERSION_MAJOR}

// Minor version of GSL 
#define GSL_VERSION_MINOR ${GSL_VERSION_MINOR}

// Patch version of GSL 
#define GSL_VERSION_PATCH ${GSL_VERSION_PATCH}

// GSL version string 
#define GSL_VERSION_STRING "${GSL_VERSION_STRING}"

#endif
