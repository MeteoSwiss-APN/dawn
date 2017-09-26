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

#ifndef DAWN_CODEGEN_CODEGEN_H
#define DAWN_CODEGEN_CODEGEN_H

#include "dawn/CodeGen/TranslationUnit.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include <memory>

namespace dawn {

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

} // namespace dawn

#endif
