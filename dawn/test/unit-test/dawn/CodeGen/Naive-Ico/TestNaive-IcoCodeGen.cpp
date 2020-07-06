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

#include "UnstructuredStencils.h"
#include "dawn/CodeGen/Options.h"
#include "dawn/Serialization/IIRSerializer.h"

#include <gtest/gtest.h>

namespace {

constexpr auto backend = dawn::codegen::Backend::CXXNaiveIco;

// NOTE: Often-changing backend. For the moment we prefer to test code generation through end-to-end
// tests checking the output. To be reconsidered once this is stable.

} // namespace