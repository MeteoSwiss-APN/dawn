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

#ifndef DAWN_CODEGEN_GRIDTOOLS_GTCODEGEN_H
#define DAWN_CODEGEN_GRIDTOOLS_GTCODEGEN_H

#include "dawn/CodeGen/CodeGen.h"
#include "dawn/Optimizer/Interval.h"
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace dawn {

class OptimizerContext;

namespace iir {
class StencilInstantiation;
class Stage;
class Stencil;
}

namespace codegen {
namespace gt {

/// @brief GridTools C++ code generation for the gridtools_clang DSL
/// @ingroup gt
class GTCodeGen : public CodeGen {
public:
  GTCodeGen(OptimizerContext* context);
  virtual ~GTCodeGen();

  virtual std::unique_ptr<TranslationUnit> generateCode() override;

  /// @brief Definitions of the gridtools::intervals
  struct IntervalDefinitions {
    IntervalDefinitions(const iir::Stencil& stencil);

    /// Intervals of the stencil
    std::unordered_set<IntervalProperties> intervalProperties_;

    /// Axis of the stencil (i.e the interval which spans accross all other intervals)
    Interval Axis;

    /// Levels of the axis
    std::set<int> Levels;

    /// Intervals of the Do-Methods of each stage
    std::unordered_map<int, std::vector<Interval>> StageIntervals;
  };

private:
  std::string generateStencilInstantiation(
      const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation);
  std::string generateGlobals(const std::shared_ptr<SIR>& Sir);
  std::string cacheWindowToString(const Cache::window& cacheWindow);
  std::string buildMakeComputation(std::vector<std::string> const& DomainMapPlaceholders,
                                   std::vector<std::string> const& makeComputation,
                                   const std::string& gridName) const;
  void
  buildPlaceholderDefinitions(MemberFunction& function,
                              std::vector<iir::Stencil::FieldInfo> const& stencilFields,
                              std::vector<std::string> const& stencilGlobalVariables,
                              std::vector<std::string> const& stencilConstructorTemplates) const;

  std::string getFieldName(std::shared_ptr<sir::Field> const& f) const { return f->Name; }

  std::string getFieldName(iir::Stencil::FieldInfo const& f) const { return f.Name; }

  bool isTemporary(std::shared_ptr<sir::Field> f) const { return f->IsTemporary; }

  bool isTemporary(iir::Stencil::FieldInfo const& f) const { return f.IsTemporary; }

  /// code generate sync methods statements for all the fields passed
  void
  generateSyncStorages(MemberFunction& method,
                       const IndexRange<std::vector<iir::Stencil::FieldInfo>>& stencilFields) const;

  /// construct a string of template parameters for storages
  std::vector<std::string> buildFieldTemplateNames(
      IndexRange<std::vector<iir::Stencil::FieldInfo>> const& stencilFields) const;

  int computeNumTemporaries(std::vector<iir::Stencil::FieldInfo> const& stencilFields) const;

  /// Maximum needed vector size of boost::fusion containers
  std::size_t mplContainerMaxSize_;
};

} // namespace gt
} // namespace codegen
} // namespace dawn

#endif
