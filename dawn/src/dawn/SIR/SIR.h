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

class FieldDimensionImpl {
public:
  std::unique_ptr<FieldDimensionImpl> clone() const { return cloneImpl(); }
  virtual ~FieldDimensionImpl() = default;
  bool operator==(const FieldDimensionImpl& other) { return equalityImpl(other); }

private:
  virtual std::unique_ptr<FieldDimensionImpl> cloneImpl() const = 0;
  virtual bool equalityImpl(const FieldDimensionImpl& other) const = 0;
};

/// @brief In the cartesian case, the horizontal dimension is an IJ-mask describing if the
/// field is allowed to have extents in I and/or J: [1,0] is a storage_i and cannot be
/// accessed with field[j+1]
///
/// @ingroup sir
class CartesianFieldDimension : public FieldDimensionImpl {
  const std::array<bool, 2> mask_;
  std::unique_ptr<FieldDimensionImpl> cloneImpl() const override {
    return std::make_unique<CartesianFieldDimension>(mask_);
  }
  virtual bool equalityImpl(const FieldDimensionImpl& other) const override {
    auto const& otherCartesian = dynamic_cast<CartesianFieldDimension const&>(other);
    return otherCartesian.I() == I() && otherCartesian.J() == J();
  }

public:
  bool I() const { return mask_[0]; }
  bool J() const { return mask_[1]; }
  explicit CartesianFieldDimension(std::array<bool, 2> mask) : mask_(mask) {}
};

/// @brief In the unstructured case, the horizontal dimension can be either dense or sparse.
/// A field on dense corresponds to 1 value for each location of type defined by the "dense location
/// type". A field on sparse corresponds to as many values as indirect neighbors defined through a
/// neighbor chain.
///
/// Construct with a neighbor chain. If it is of size = 1, then the dimension is dense (with
/// location type = single element of chain), otherwise sparse (with dense location type being the
/// first element of the chain).
///
/// @ingroup sir
class UnstructuredFieldDimension : public FieldDimensionImpl {
  std::unique_ptr<FieldDimensionImpl> cloneImpl() const override {
    return std::make_unique<UnstructuredFieldDimension>(iterSpace_.Chain, iterSpace_.IncludeCenter);
  }
  virtual bool equalityImpl(const FieldDimensionImpl& other) const override {
    auto const& otherUnstructured = dynamic_cast<UnstructuredFieldDimension const&>(other);
    return iterSpace_ == otherUnstructured.iterSpace_;
  }
  const ast::UnstructuredIterationSpace iterSpace_;

public:
  explicit UnstructuredFieldDimension(ast::NeighborChain neighborChain, bool includeCenter = false);
  /// @brief Returns the neighbor chain encoding the sparse part (isSparse() must be true!).
  const ast::NeighborChain& getNeighborChain() const;
  /// @brief Returns the dense location (always present)
  ast::LocationType getDenseLocationType() const { return iterSpace_.Chain[0]; }
  /// @brief Returns the last sparse location type if there is a sparse part, otherwise returns the
  /// dense part.
  ast::LocationType getLastSparseLocationType() const { return iterSpace_.Chain.back(); }
  bool isSparse() const { return iterSpace_.Chain.size() > 1; }
  bool isDense() const { return !isSparse(); }
  bool getIncludeCenter() const { return iterSpace_.IncludeCenter; }
  ast::UnstructuredIterationSpace getIterSpace() const { return iterSpace_; }
  std::string toString() const;
};

class HorizontalFieldDimension {
  std::unique_ptr<FieldDimensionImpl> impl_;

public:
  // Construct a Cartesian horizontal field dimension with specified ij mask.
  HorizontalFieldDimension(dawn::ast::cartesian_, std::array<bool, 2> mask)
      : impl_(std::make_unique<CartesianFieldDimension>(mask)) {}

  // Construct a Unstructured horizontal field sparse dimension with specified neighbor chain
  // (sparse part). Dense part is the first element of the chain.
  HorizontalFieldDimension(dawn::ast::unstructured_, ast::NeighborChain neighborChain,
                           bool includeCenter = false)
      : impl_(std::make_unique<UnstructuredFieldDimension>(neighborChain, includeCenter)) {}
  // Construct a Unstructured horizontal field dense dimension with specified (dense) location type.
  HorizontalFieldDimension(dawn::ast::unstructured_, ast::LocationType locationType,
                           bool includeCenter = false)
      : impl_(std::make_unique<UnstructuredFieldDimension>(ast::NeighborChain{locationType},
                                                           includeCenter)) {}

  HorizontalFieldDimension(const HorizontalFieldDimension& other) { *this = other; }
  HorizontalFieldDimension(HorizontalFieldDimension&& other) { *this = other; };

  HorizontalFieldDimension& operator=(const HorizontalFieldDimension& other) {
    impl_ = other.impl_->clone();
    return *this;
  }
  HorizontalFieldDimension& operator=(HorizontalFieldDimension&& other) {
    impl_ = std::move(other.impl_);
    return *this;
  }

  bool operator==(const HorizontalFieldDimension& other) const { return *impl_ == *other.impl_; }

  ast::GridType getType() const;

  template <typename T>
  friend T dimension_cast(HorizontalFieldDimension const& dimension);
  template <typename T>
  friend bool dimension_isa(HorizontalFieldDimension const& dimension);
};

template <typename T>
T dimension_cast(HorizontalFieldDimension const& dimension) {
  using PlainT = std::remove_reference_t<T>;
  static_assert(std::is_base_of_v<FieldDimensionImpl, PlainT>,
                "Can only be casted to a valid field dimension implementation");
  static_assert(std::is_const_v<PlainT>, "Can only be casted to const");
  return *dynamic_cast<std::add_pointer_t<T>>(dimension.impl_.get());
}

template <typename T>
bool dimension_isa(HorizontalFieldDimension const& dimension) {
  using PlainT = std::remove_pointer_t<std::remove_reference_t<T>>;
  static_assert(std::is_base_of_v<FieldDimensionImpl, PlainT>,
                "Can only be casted to a valid field dimension implementation");
  return static_cast<bool>(dynamic_cast<PlainT*>(dimension.impl_.get()));
}

class FieldDimensions {
public:
  FieldDimensions(HorizontalFieldDimension&& horizontalFieldDimension, bool maskK)
      : horizontalFieldDimension_(horizontalFieldDimension), maskK_(maskK) {
    if(!maskK && dimension_isa<CartesianFieldDimension>(*horizontalFieldDimension_)) {
      auto cartDims = dimension_cast<const CartesianFieldDimension&>(*horizontalFieldDimension_);
      DAWN_ASSERT_MSG(cartDims.I() || cartDims.J(),
                      "a field cant' have all dimensions masked out!");
    }
  }
  FieldDimensions(bool maskK) : maskK_(maskK) {
    DAWN_ASSERT_MSG(
        maskK_, "a field can't have null horizontal dimensions as well as masked out k dimension!");
  }
  FieldDimensions(const FieldDimensions&) = default;
  FieldDimensions(FieldDimensions&&) = default;

  FieldDimensions& operator=(const FieldDimensions&) = default;
  FieldDimensions& operator=(FieldDimensions&&) = default;

  bool operator==(const FieldDimensions& other) const {
    return (maskK_ == other.maskK_ && horizontalFieldDimension_ == other.horizontalFieldDimension_);
  }

  bool K() const { return maskK_; }
  const HorizontalFieldDimension& getHorizontalFieldDimension() const {
    DAWN_ASSERT_MSG(!isVertical(), "attempted to get horizontal dimension of a vertical field!");
    return horizontalFieldDimension_.value();
  }
  bool isVertical() const { return !horizontalFieldDimension_.has_value(); }
  std::string toString() const;

  // returns number of dimensions (1-3)
  int numSpatialDimensions() const;

  // returns the rank of the corresponding storage (multidimensional array)
  int rank() const;

private:
  std::optional<HorizontalFieldDimension> horizontalFieldDimension_;
  bool maskK_;
};

/// @brief Representation of a field
/// @ingroup sir
struct Field : public StencilFunctionArg {
  Field(const std::string& name, FieldDimensions&& fieldDimensions,
        SourceLocation loc = SourceLocation())
      : StencilFunctionArg{name, ArgumentKind::Field, loc}, IsTemporary(false),
        Dimensions(fieldDimensions) {}

  bool IsTemporary;
  FieldDimensions Dimensions;

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
