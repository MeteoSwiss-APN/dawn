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
#include "dawn/CodeGen/CodeGenProperties.h"
#include "dawn/CodeGen/TranslationUnit.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/DiagnosticsEngine.h"
#include "dawn/Support/IndexRange.h"
#include <memory>

namespace dawn {
namespace codegen {

template <typename Key, typename Value>
extern std::map<Key, Value> orderMap(const std::unordered_map<Key, Value>& umap) {
  std::map<Key, Value> m;
  for(const auto& f : umap)
    m.insert(f);

  return m;
}

/// @brief Interface of the backend code generation
/// @ingroup codegen
class CodeGen {
protected:
  std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>& context;
  DiagnosticsEngine& diagEngine;
  struct codeGenOption {
    int MaxHaloPoints;
  } codeGenOptions;

  static size_t getVerticalTmpHaloSize(iir::Stencil const& stencil);
  size_t getVerticalTmpHaloSizeForMultipleStencils(
      const std::vector<std::unique_ptr<iir::Stencil>>& stencils) const;
  virtual void addTempStorageTypedef(Structure& stencilClass, iir::Stencil const& stencil) const;
  void addTmpStorageDeclaration(
      Structure& stencilClass,
      IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& tmpFields) const;
  virtual void
  addTmpStorageInit(MemberFunction& ctr, const iir::Stencil& stencil,
                    IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& tempFields) const;
  void
  addTmpStorageInitStencilWrapperCtr(MemberFunction& ctr,
                                     const std::vector<std::unique_ptr<iir::Stencil>>& stencils,
                                     const std::vector<std::string>& tempFields) const;

  void addBCFieldInitStencilWrapperCtr(MemberFunction& ctr,
                                       const CodeGenProperties& codeGenProperties) const;
  void generateBCFieldMembers(Class& stencilWrapperClass,
                              const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation,
                              const CodeGenProperties& codeGenProperties) const;

  void addMplIfdefs(std::vector<std::string>& ppDefines, int mplContainerMaxSize) const;

  const std::string tmpStorageTypename_ = "tmp_storage_t";
  const std::string tmpMetadataTypename_ = "tmp_meta_data_t";
  const std::string tmpMetadataName_ = "m_tmp_meta_data";
  const std::string tmpStorageName_ = "m_tmp_storage";
  const std::string bigWrapperMetadata_ = "m_meta_data";

public:
  CodeGen(std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>& ctx,
          DiagnosticsEngine& engine, int maxHaloPoints)
      : context(ctx), diagEngine(engine), codeGenOptions{maxHaloPoints} {};
  virtual ~CodeGen() {}

  /// @brief Generate code
  virtual std::unique_ptr<TranslationUnit> generateCode() = 0;

  static std::string getStorageType(const sir::Field& field);
  static std::string getStorageType(const iir::Stencil::FieldInfo& field);
  static std::string getStorageType(Array3i dimensions);

  void generateBoundaryConditionFunctions(
      Class& stencilWrapperClass,
      const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation) const;

  CodeGenProperties
  computeCodeGenProperties(const iir::StencilInstantiation* stencilInstantiation) const;

  virtual void generateGlobalsAPI(const iir::StencilInstantiation& stencilInstantiation,
                                  Class& stencilWrapperClass,
                                  const sir::GlobalVariableMap& globalsMap,
                                  const CodeGenProperties& codeGenProperties) const;
  virtual std::string generateGlobals(const sir::GlobalVariableMap& globalsMaps,
                                      std::string namespace_) const;
  void generateBCHeaders(std::vector<std::string>& ppDefines) const;
};

} // namespace codegen
} // namespace dawn

#endif
