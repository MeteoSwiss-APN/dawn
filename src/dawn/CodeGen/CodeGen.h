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

#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/CodeGen/TranslationUnit.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Support/IndexRange.h"
#include <memory>

namespace dawn {
namespace codegen {

/// @brief Interface of the backend code generation
/// @ingroup codegen
class CodeGen {
protected:
  OptimizerContext* context_;

  static size_t getVerticalTmpHaloSize(Stencil const& stencil);
  size_t getVerticalTmpHaloSizeForMultipleStencils(
      const std::vector<std::shared_ptr<Stencil>>& stencils) const;
  void addTempStorageTypedef(Structure& stencilClass, Stencil const& stencil) const;
  void addTmpStorageDeclaration(
      Structure& stencilClass,
      IndexRange<const std::vector<dawn::Stencil::FieldInfo>>& tmpFields) const;
  void addTmpStorageInit(MemberFunction& ctr, const Stencil& stencil,
                         IndexRange<const std::vector<dawn::Stencil::FieldInfo>>& tempFields) const;
  void addTmpStorageInit_wrapper(MemberFunction& ctr,
                                 const std::vector<std::shared_ptr<Stencil>>& stencils,
                                 const std::vector<std::string>& tempFields) const;

  const std::string tmpStorageTypename_ = "tmp_storage_t";
  const std::string tmpMetadataTypename_ = "tmp_meta_data_t";
  const std::string tmpMetadataName_ = "m_tmp_meta_data";
  const std::string tmpStorageName_ = "m_tmp_storage";
  const std::string bigWrapperMetadata_ = "m_meta_data";

public:
  CodeGen(OptimizerContext* context) : context_(context){};
  virtual ~CodeGen() {}

  /// @brief Generate code
  virtual std::unique_ptr<TranslationUnit> generateCode() = 0;

  /// @brief Get the optimizer context
  const OptimizerContext* getOptimizerContext() const { return context_; }
};

} // namespace codegen
} // namespace dawn

#endif
