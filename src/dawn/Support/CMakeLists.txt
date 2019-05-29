##===------------------------------------------------------------------------------*- CMake -*-===##
##                          _                      
##                         | |                     
##                       __| | __ ___      ___ ___  
##                      / _` |/ _` \ \ /\ / / '_  | 
##                     | (_| | (_| |\ V  V /| | | |
##                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
##
##
##  This file is distributed under the MIT License (MIT). 
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

yoda_configure_file(Config.h.cmake)

yoda_add_library(
  NAME DawnSupport
  SOURCES AlignOf.h
          Array.cpp
          Array.h
          ArrayRef.h
          Assert.cpp
          Assert.h
          Casting.h
          Compiler.h
          Config.h.cmake
          DoubleSidedMap.h
          EditDistance.h
          FileUtil.cpp
          FileUtil.h
          Format.h
          HashCombine.h
          IndexGenerator.cpp
          IndexGenerator.h
          IndexRange.h
          Json.h
          Logging.cpp
          Logging.h
          MathExtras.h
          NonCopyable.h
          Printing.h          
          RemoveIf.hpp
          SmallString.h
          SmallVector.cpp
          SmallVector.h
          SourceLocation.cpp
          SourceLocation.h
          STLExtras.h
          StringRef.cpp
          StringRef.h
          StringSwitch.h
          StringUtil.cpp
          StringUtil.h
          Twine.cpp
          Twine.h
          Type.cpp
          Type.h
          TypeTraits.h
          UIDGenerator.cpp
          UIDGenerator.h
          Unreachable.cpp
          Unreachable.h
          VisitorHelpers.h
          ../Dawn.h
  OBJECT
)

yoda_combine_libraries(
  NAME DawnSupport
  OBJECTS DawnSupportObjects
  INSTALL_DESTINATION ${DAWN_INSTALL_LIB_DIR}
  VERSION ${DAWN_VERSION}
  EXPORT_GROUP DawnTargets
)

