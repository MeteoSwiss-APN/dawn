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
#include "dawn/CodeGen/CodeGenProperties.h"
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

  static size_t getVerticalTmpHaloSize(iir::Stencil const& stencil);
  size_t getVerticalTmpHaloSizeForMultipleStencils(
      const std::vector<std::unique_ptr<iir::Stencil>>& stencils) const;
  virtual void addTempStorageTypedef(Structure& stencilClass, iir::Stencil const& stencil) const;
  void addTmpStorageDeclaration(
      Structure& stencilClass,
      IndexRange<const std::unordered_map<int, iir::Stencil::FieldInfo>>& tmpFields) const;
  virtual void addTmpStorageInit(
      MemberFunction& ctr, const iir::Stencil& stencil,
      IndexRange<const std::unordered_map<int, iir::Stencil::FieldInfo>>& tempFields) const;
  void
  addTmpStorageInitStencilWrapperCtr(MemberFunction& ctr,
                                     const std::vector<std::unique_ptr<iir::Stencil>>& stencils,
                                     const std::vector<std::string>& tempFields) const;

  void addMplIfdefs(std::vector<std::string>& ppDefines, int mplContainerMaxSize,
                    int MaxHaloPoints) const;

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

  static std::string getStorageType(const sir::Field& field);
  static std::string getStorageType(const iir::Stencil::FieldInfo& field);
  static std::string getStorageType(Array3i dimensions);

  virtual void generateGlobalsAPI(const iir::StencilInstantiation& stencilInstantiation,
                                  Class& stencilWrapperClass,
                                  const sir::GlobalVariableMap& globalsMap,
                                  const CodeGenProperties& codeGenProperties) const;
  virtual std::string generateGlobals(std::shared_ptr<SIR> const& sir,
                                      std::string namespace_) const;
};

} // namespace codegen
} // namespace dawn

#endif
