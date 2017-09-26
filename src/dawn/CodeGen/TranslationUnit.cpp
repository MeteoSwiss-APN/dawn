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

#include "dawn/CodeGen/TranslationUnit.h"

namespace dawn {

TranslationUnit::TranslationUnit(std::string filename, std::vector<std::string>&& ppDefines,
                                 std::map<std::string, std::string>&& stencils,
                                 std::string&& globals)
    : filename_(std::move(filename)), ppDefines_(std::move(ppDefines)),
      globals_(std::move(globals)), stencils_(std::move(stencils)) {}

} // namespace dawn
