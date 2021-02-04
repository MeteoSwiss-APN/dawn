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

#include "dawn/AST/Attr.h"
#include "dawn/AST/FieldDimension.h"
#include "dawn/AST/GridType.h"
#include "dawn/AST/Interval.h"
#include "dawn/AST/IterationSpace.h"
#include "dawn/AST/Value.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/VerticalRegion.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/ComparisonHelpers.h"
#include "dawn/Support/NonCopyable.h"
#include "dawn/Support/SourceLocation.h"

namespace dawn {

/// @namespace sir
/// @brief This namespace contains a C++ implementation of the SIR specification
/// @ingroup sir
namespace sir {

//===------------------------------------------------------------------------------------------===//
//     StencilFunctionArgs (Field, Direction and Offset)
//===------------------------------------------------------------------------------------------===//

/// @brief Base class of objects which can be used as arguments in StencilFunctions
/// @ingroup sir
struct StencilFunctionArg {
  enum class ArgumentKind { Field, Direction, Offset };

  static constexpr int NumArgTypes = 3;

  std::string Name;   ///< Name of the argument
  ArgumentKind Kind;  ///< Type of argument
  SourceLocation Loc; ///< Source location

  bool operator==(const StencilFunctionArg& rhs) const;

  CompareResult comparison(const sir::StencilFunctionArg& rhs) const;
};

/// @brief Representation of a field
/// @ingroup sir
struct Field : public StencilFunctionArg {
  Field(const std::string& name, ast::FieldDimensions&& fieldDimensions,
        SourceLocation loc = SourceLocation())
      : StencilFunctionArg{name, ArgumentKind::Field, loc}, IsTemporary(false),
        Dimensions(fieldDimensions) {}

  bool IsTemporary;
  ast::FieldDimensions Dimensions;

  static bool classof(const StencilFunctionArg* arg) { return arg->Kind == ArgumentKind::Field; }
  bool operator==(const Field& rhs) const { return comparison(rhs); }

  CompareResult comparison(const Field& rhs) const;
};

/// @brief Representation of a direction (e.g `i`)
/// @ingroup sir
struct Direction : public StencilFunctionArg {
  Direction(const std::string& name, SourceLocation loc = SourceLocation())
      : StencilFunctionArg{name, ArgumentKind::Direction, loc} {}

  static bool classof(const StencilFunctionArg* arg) {
    return arg->Kind == ArgumentKind::Direction;
  }
};

/// @brief Representation of an Offset (e.g `i + 1`)
/// @ingroup sir
struct Offset : public StencilFunctionArg {
  Offset(const std::string& name, SourceLocation loc = SourceLocation())
      : StencilFunctionArg{name, ArgumentKind::Offset, loc} {}

  static bool classof(const StencilFunctionArg* arg) { return arg->Kind == ArgumentKind::Offset; }
};

//===------------------------------------------------------------------------------------------===//
//     StencilFunction
//===------------------------------------------------------------------------------------------===//

/// @brief Representation of a stencil function
/// @ingroup sir
struct StencilFunction {
  std::string Name;                                      ///< Name of the stencil function
  SourceLocation Loc;                                    ///< Source location of the stencil func
  std::vector<std::shared_ptr<StencilFunctionArg>> Args; ///< Arguments of the stencil function
  std::vector<std::shared_ptr<ast::Interval>> Intervals; ///< Vertical intervals of the specializations
  std::vector<std::shared_ptr<ast::AST>> Asts;      ///< ASTs of the specializations
  ast::Attr Attributes;                                  ///< Attributes of the stencil function

  /// @brief Check if the Stencil function contains specializations
  ///
  /// If `Intervals` is empty and `Asts` contains one element, the StencilFunction is not
  /// specialized.
  bool isSpecialized() const;

  /// @brief Get the AST of the specified vertical interval or `NULL` if the function is not
  /// specialized for this interval
  std::shared_ptr<ast::AST> getASTOfInterval(const ast::Interval& interval) const;

  bool operator==(const sir::StencilFunction& rhs) const;
  CompareResult comparison(const StencilFunction& rhs) const;

  bool hasArg(std::string name) {
    return std::find_if(Args.begin(), Args.end(),
                        [&](std::shared_ptr<sir::StencilFunctionArg> arg) {
                          return name == arg->Name;
                        }) != Args.end();
  }
};

//===------------------------------------------------------------------------------------------===//
//     Stencil
//===------------------------------------------------------------------------------------------===//

/// @brief Representation of a stencil which is a sequence of calls to other stencils
/// (`StencilCall`) or vertical regions (`VerticalRegion`)
/// @ingroup sir
struct Stencil : public dawn::NonCopyable {
  Stencil();

  std::string Name;                           ///< Name of the stencil
  SourceLocation Loc;                         ///< Source location of the stencil declaration
  std::shared_ptr<ast::AST> StencilDescAst;   ///< Stencil description AST
  std::vector<std::shared_ptr<Field>> Fields; ///< Fields referenced by this stencil
  ast::Attr Attributes;                            ///< Attributes of the stencil

  bool operator==(const Stencil& rhs) const;
  CompareResult comparison(const Stencil& rhs) const;
};

} // namespace sir

//===------------------------------------------------------------------------------------------===//
//     SIR
//===------------------------------------------------------------------------------------------===//

/// @brief Definition of the Stencil Intermediate Representation (SIR)
/// @ingroup sir
struct SIR : public dawn::NonCopyable {

  /// @brief Default Ctor that initializes all the shared pointers
  SIR(const ast::GridType gridType);

  /// @brief Dump the SIR to stdout
  void dump(std::ostream& os);

  /// @brief Compares two SIRs for equality in contents
  ///
  /// The `Filename` as well as the SourceLocations are not taken into account.
  bool operator==(const SIR& rhs) const;
  bool operator!=(const SIR& rhs) const;

  /// @brief Compares two SIRs for equality in contents
  ///
  /// The `Filename` as well as the SourceLocations and Attributes are not taken into account.
  CompareResult comparison(const SIR& rhs) const;

  /// @brief Dump SIR to the given stream
  friend std::ostream& operator<<(std::ostream& os, const SIR& Sir);

  std::string Filename;                                ///< Name of the file the SIR was parsed from
  std::vector<std::shared_ptr<sir::Stencil>> Stencils; ///< List of stencils
  std::vector<std::shared_ptr<sir::StencilFunction>> StencilFunctions; ///< List of stencil function
  std::shared_ptr<ast::GlobalVariableMap> GlobalVariableMap;           ///< Map of global variables
  const ast::GridType GridType;
};

} // namespace dawn
