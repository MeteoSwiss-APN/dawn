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

#pragma once

#include "dawn/CodeGen/CodeGen.h"
#include "dawn/CodeGen/CodeGenProperties.h"
#include "dawn/CodeGen/Options.h"
#include "dawn/IIR/Interval.h"
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace dawn {

namespace iir {
class StencilInstantiation;
class Stage;
class Stencil;
} // namespace iir

namespace codegen {
namespace gt {

/// @brief Run the GridTools code generation
std::unique_ptr<TranslationUnit>
run(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
        stencilInstantiationMap,
    const Options& options = {});

/// @brief GridTools C++ code generation for the gtclang DSL
/// @ingroup gt
class GTCodeGen : public CodeGen {
public:
  GTCodeGen(const StencilInstantiationContext& ctx, bool useParallelEP, int maxHaloPoints,
            bool runWithSync = true);
  virtual ~GTCodeGen();

  virtual std::unique_ptr<TranslationUnit> generateCode() override;

  /// @brief Definitions of the gridtools::intervals
  struct IntervalDefinitions {
    IntervalDefinitions(const iir::Stencil& stencil);

    /// Intervals of the stencil
    std::unordered_set<iir::IntervalProperties> intervalProperties_;

    /// Axis of the stencil (i.e the interval which spans accross all other intervals)
    iir::Interval Axis;

    /// Levels of the axis
    std::set<int> Levels;

    /// Intervals of the Do-Methods of each stage
    std::unordered_map<int, std::vector<iir::Interval>> StageIntervals;

    // TODO we should compute the OffsetLimit, not use a hard-coded value!
    static constexpr int OffsetLimit = 3;

    // TODO we should avoid the ExtraOffsets, not use a hard-coded value!
    static constexpr int ExtraOffsets = 1;
  };

private:
  std::string generateStencilInstantiation(
      const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation);
  std::string cacheWindowToString(const iir::Cache::window& cacheWindow);

  void generatePlaceholderDefinitions(
      Structure& function, const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
      const sir::GlobalVariableMap& globalsMap, const CodeGenProperties& codeGenProperties) const;

  std::string getFieldName(std::shared_ptr<sir::Field> const& f) const { return f->Name; }

  std::string getFieldName(iir::Stencil::FieldInfo const& f) const { return f.Name; }

  bool isTemporary(std::shared_ptr<sir::Field> f) const { return f->IsTemporary; }

  bool isTemporary(iir::Stencil::FieldInfo const& f) const { return f.IsTemporary; }

  void generateGlobalsAPI(const iir::StencilInstantiation& stencilInstantiation,
                          Class& stencilWrapperClass, const sir::GlobalVariableMap& globalsMap,
                          const CodeGenProperties& codeGenProperties) const override;

  void generateStencilWrapperMembers(
      Class& stencilWrapperClass,
      const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation,
      CodeGenProperties& codeGenProperties);

  void
  generateStencilWrapperCtr(Class& stencilWrapperClass,
                            const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                            CodeGenProperties& codeGenProperties) const;

  void
  generateStencilWrapperRun(Class& stencilWrapperClass,
                            const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation,
                            const CodeGenProperties& codeGenProperties) const;

  void
  generateStencilWrapperPublicMemberFunctions(Class& stencilWrapperClass,
                                              const CodeGenProperties& codeGenProperties) const;

  void
  generateStencilClasses(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                         Class& stencilWrapperClass, CodeGenProperties& codeGenProperties);

  void generateGridConstruction(MemberFunction& stencilConstructor, const iir::Stencil& stencil,
                                IntervalDefinitions& intervalDefinitions,
                                const CodeGenProperties& codeGenProperties) const;

  static std::string getAxisName(const std::string& stencilName);
  static std::string getGridName(const std::string& stencilName);

  /// construct a string of template parameters for storages
  std::vector<std::string> buildFieldTemplateNames(
      IndexRange<std::vector<iir::Stencil::FieldInfo>> const& stencilFields) const;

  /// Maximum needed vector size of boost::fusion containers
  std::size_t mplContainerMaxSize_;

  /// Use the parallel keyword for mulistages
  struct GTCodeGenOptions {
    bool useParallelEP_;
    bool runWithSync_;
  } codeGenOptions_;
};

} // namespace gt
} // namespace codegen
} // namespace dawn
