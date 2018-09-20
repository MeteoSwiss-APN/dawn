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

#ifndef DAWN_CODEGEN_CXXNAIVE_CXXNAIVECODEGEN_H
#define DAWN_CODEGEN_CXXNAIVE_CXXNAIVECODEGEN_H

#include "dawn/CodeGen/CodeGen.h"
#include "dawn/CodeGen/CodeGenProperties.h"
#include "dawn/IIR/Interval.h"
#include "dawn/Support/IndexRange.h"
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dawn {
namespace iir {
class StencilInstantiation;
}

class OptimizerContext;

namespace codegen {
namespace cxxnaive {

/// @brief GridTools C++ code generation for the gridtools_clang DSL
/// @ingroup cxxnaive
class CXXNaiveCodeGen : public CodeGen {
public:
  ///@brief constructor
  CXXNaiveCodeGen(OptimizerContext* context);
  virtual ~CXXNaiveCodeGen();
  virtual std::unique_ptr<TranslationUnit> generateCode() override;

private:
  std::string generateStencilInstantiation(const iir::StencilInstantiation* stencilInstantiation);
};
} // namespace cxxnaive
} // namespace codegen
} // namespace dawn

#endif
