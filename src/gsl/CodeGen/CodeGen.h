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

#ifndef GSL_CODEGEN_CODEGEN_H
#define GSL_CODEGEN_CODEGEN_H

#include "gsl/CodeGen/TranslationUnit.h"
#include "gsl/Optimizer/OptimizerContext.h"
#include <memory>

namespace gsl {

/// @brief Interface of the backend code generation
/// @ingroup codegen
class CodeGen {
protected:
  OptimizerContext* context_;

public:
  CodeGen(OptimizerContext* context) : context_(context){};
  virtual ~CodeGen() {}

  /// @brief Generate code
  virtual std::unique_ptr<TranslationUnit> generateCode() = 0;

  /// @brief Get the optimizer context
  const OptimizerContext* getOptimizerContext() const { return context_; }
};

} // namespace gsl

#endif
