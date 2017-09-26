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

#ifndef DAWN_CODEGEN_GTCLANGNAIVECXXCODEGEN_H
#define DAWN_CODEGEN_GTCLANGNAIVECXXCODEGEN_H

#include "dawn/CodeGen/CodeGen.h"
#include "dawn/Optimizer/Interval.h"
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace dawn {

class StencilInstantiation;
class OptimizerContext;

/// @brief GridTools C++ code generation for the gridtools_clang DSL
/// @ingroup codegen
class GTClangNaiveCXXCodeGen : public CodeGen {
public:
  GTClangNaiveCXXCodeGen(OptimizerContext* context);
  virtual ~GTClangNaiveCXXCodeGen();
  virtual std::unique_ptr<TranslationUnit> generateCode() override;

private:
  std::string generateStencilInstantiation(const StencilInstantiation* stencilInstantiation);
};

} // namespace dawn

#endif
