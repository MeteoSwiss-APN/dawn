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

#ifndef GSL_CODEGEN_GTCLANGNAIVECXXCODEGEN_H
#define GSL_CODEGEN_GTCLANGNAIVECXXCODEGEN_H

#include "gsl/CodeGen/CodeGen.h"
#include "gsl/Optimizer/Interval.h"
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace gsl {

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

} // namespace gsl

#endif
