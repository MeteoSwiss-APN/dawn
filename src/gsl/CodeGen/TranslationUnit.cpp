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

#include "gsl/CodeGen/TranslationUnit.h"

namespace gsl {

TranslationUnit::TranslationUnit(std::string filename, std::vector<std::string>&& ppDefines,
                                 std::map<std::string, std::string>&& stencils,
                                 std::string&& globals)
    : filename_(std::move(filename)), ppDefines_(std::move(ppDefines)),
      globals_(std::move(globals)), stencils_(std::move(stencils)) {}

} // namespace gsl
