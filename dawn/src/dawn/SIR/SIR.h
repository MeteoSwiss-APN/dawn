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

#ifndef DAWN_SIR_SIR_H
#define DAWN_SIR_SIR_H

#include "dawn/AST/GridType.h"
#include "dawn/AST/Tags.h"
#include "dawn/SIR/AST.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/ComparisonHelpers.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/HashCombine.h"
#include "dawn/Support/Json.h"
#include "dawn/Support/NonCopyable.h"
#include "dawn/Support/SourceLocation.h"
#include "dawn/Support/Type.h"
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace dawn {

/// @namespace sir
/// @brief This namespace contains a C++ implementation of the SIR specification
/// @ingroup sir
namespace sir {

/// @brief Attributes attached to various SIR objects which allow to change the behavior on per
/// stencil basis
/// @ingroup sir
class Attr {
  unsigned attrBits_;

public:
  Attr() : attrBits_(0) {}

  /// @brief Attribute bit-mask
  enum class Kind : unsigned {
    NoCodeGen = 1 << 0,        ///< Don't generate code for this stencil
    MergeStages = 1 << 1,      ///< Merge the Stages of this stencil
    MergeDoMethods = 1 << 2,   ///< Merge the Do-Methods of this stencil
    MergeTemporaries = 1 << 3, ///< Merge the temporaries of this stencil
    UseKCaches = 1 << 4        ///< Use K-Caches
  };

  /// @brief Check if `attr` bit is set
  bool has(Kind attr) const { return (attrBits_ >> static_cast<unsigned>(attr)) & 1; }

  /// @brief Check if any of the `attrs` bits is set
  /// @{
  bool hasOneOf(Kind attr1, Kind attr2) const { return has(attr1) || has(attr2); }

  template <typename... AttrTypes>
  bool hasOneOf(Kind attr1, Kind attr2, AttrTypes... attrs) const {
    return has(attr1) || hasOneOf(attr2, attrs...);
  }
  /// @}

  ///@brief getting the Bits
  unsigned getBits() const { return attrBits_; }
  /// @brief Set `attr`bit
  void set(Kind attr) { attrBits_ |= 1 << static_cast<unsigned>(attr); }

  /// @brief Unset `attr` bit
  void unset(Kind attr) { attrBits_ &= ~(1 << static_cast<unsigned>(attr)); }

  /// @brief Clear all attributes
  void clear() { attrBits_ = 0; }

  bool operator==(const Attr& rhs) const { return getBits() == rhs.getBits(); }
  bool operator!=(const Attr& rhs) const { return getBits() != rhs.getBits(); }
};

//===------------------------------------------------------------------------------------------===//
//     Interval
//===------------------------------------------------------------------------------------------===//

/// @brief Representation of a vertical interval, given by a lower and upper bound where a bound
/// is represented by a level and an offset (`bound = level + offset`)
///
/// The Interval has to satisfy the following invariants:
///  - `lowerLevel >= Interval::Start`
///  - `upperLevel <= Interval::End`
///  - `(lowerLevel + lowerOffset) <= (upperLevel + upperOffset)`
///
/// @ingroup sir
struct Interval {
  enum LevelKind : int { Start = 0, End = (1 << 20) };

  Interval(int lowerLevel, int upperLevel, int lowerOffset = 0, int upperOffset = 0)
      : LowerLevel(lowerLevel), UpperLevel(upperLevel), LowerOffset(lowerOffset),
        UpperOffset(upperOffset) {
    DAWN_ASSERT(lowerLevel >= LevelKind::Start && upperLevel <= LevelKind::End);
    DAWN_ASSERT((lowerLevel + lowerOffset) <= (upperLevel + upperOffset));
  }

  int LowerLevel;
  int UpperLevel;
  int LowerOffset;
  int UpperOffset;

  /// @name Comparison operator
  /// @{
  bool operator==(const Interval& other) const {
    return LowerLevel == other.LowerLevel && UpperLevel == other.UpperLevel &&
           LowerOffset == other.LowerOffset && UpperOffset == other.UpperOffset;
  }
  bool operator!=(const Interval& other) const { return !(*this == other); }

  CompareResult comparison(const Interval& rhs) const;
  /// @}

  /// @brief Convert to string
  /// @{
  std::string toString() const;
  friend std::ostream& operator<<(std::ostream& os, const Interval& interval);
  /// @}
};

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
    return std::make_unique<UnstructuredFieldDimension>(neighborChain_);
  }
  virtual bool equalityImpl(const FieldDimensionImpl& other) const override {
    auto const& otherUnstructured = dynamic_cast<UnstructuredFieldDimension const&>(other);
    return std::equal(neighborChain_.begin(), neighborChain_.end(),
                      otherUnstructured.neighborChain_.begin());
  }

  const ast::NeighborChain neighborChain_;

public:
  explicit UnstructuredFieldDimension(const ast::NeighborChain neighborChain);
  /// @brief Returns the neighbor chain encoding the sparse part (isSparse() must be true!).
  const ast::NeighborChain& getNeighborChain() const;
  /// @brief Returns the dense location (always present)
  ast::LocationType getDenseLocationType() const { return neighborChain_[0]; }
  /// @brief Returns the last sparse location type if there is a sparse part, otherwise returns the
  /// dense part.
  ast::LocationType getLastSparseLocationType() const { return neighborChain_.back(); }
  bool isSparse() const { return neighborChain_.size() > 1; }
  bool isDense() const { return !isSparse(); }
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
  HorizontalFieldDimension(dawn::ast::unstructured_, ast::NeighborChain neighborChain)
      : impl_(std::make_unique<UnstructuredFieldDimension>(neighborChain)) {}
  // Construct a Unstructured horizontal field dense dimension with specified (dense) location type.
  HorizontalFieldDimension(dawn::ast::unstructured_, ast::LocationType locationType)
      : impl_(std::make_unique<UnstructuredFieldDimension>(ast::NeighborChain{locationType})) {}

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

class FieldDimensions {
public:
  FieldDimensions(HorizontalFieldDimension&& horizontalFieldDimension, bool maskK)
      : horizontalFieldDimension_(horizontalFieldDimension), maskK_(maskK) {}
  FieldDimensions(const FieldDimensions&) = default;
  FieldDimensions(FieldDimensions&&) = default;

  FieldDimensions& operator=(const FieldDimensions&) = default;
  FieldDimensions& operator=(FieldDimensions&&) = default;

  bool operator==(const FieldDimensions& other) const {
    return (maskK_ == other.maskK_ && horizontalFieldDimension_ == other.horizontalFieldDimension_);
  }

  bool K() const { return maskK_; }
  const HorizontalFieldDimension& getHorizontalFieldDimension() const {
    return horizontalFieldDimension_;
  }
  std::string toString() const;

private:
  HorizontalFieldDimension horizontalFieldDimension_;
  bool maskK_;
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
  return (bool)(dynamic_cast<PlainT*>(dimension.impl_.get()));
}

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
  std::vector<std::shared_ptr<Interval>> Intervals; ///< Vertical intervals of the specializations
  std::vector<std::shared_ptr<sir::AST>> Asts;      ///< ASTs of the specializations
  Attr Attributes;                                  ///< Attributes of the stencil function

  /// @brief Check if the Stencil function contains specializations
  ///
  /// If `Intervals` is empty and `Asts` contains one element, the StencilFunction is not
  /// specialized.
  bool isSpecialized() const;

  /// @brief Get the AST of the specified vertical interval or `NULL` if the function is not
  /// specialized for this interval
  std::shared_ptr<sir::AST> getASTOfInterval(const Interval& interval) const;

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
//     StencilDescription
//===------------------------------------------------------------------------------------------===//

/// @brief A vertical region is given by a list of statements (given as an AST) executed on a
/// specific vertical interval in a given loop order
/// @ingroup sir
struct VerticalRegion {
  enum class LoopOrderKind { Forward, Backward };

  SourceLocation Loc;                         ///< Source location of the vertical region
  std::shared_ptr<sir::AST> Ast;              ///< AST of the region
  std::shared_ptr<Interval> VerticalInterval; ///< Interval description of the region
  LoopOrderKind LoopOrder;                    ///< Loop order (usually associated with the k-loop)

  /// If it is not instantiated, iteration over the full domain is assumed.
  std::array<std::optional<Interval>, 2> IterationSpace; /// < Iteration space in the horizontal.

  VerticalRegion(const std::shared_ptr<sir::AST>& ast,
                 const std::shared_ptr<Interval>& verticalInterval, LoopOrderKind loopOrder,
                 SourceLocation loc = SourceLocation())
      : Loc(loc), Ast(ast), VerticalInterval(verticalInterval), LoopOrder(loopOrder) {}
  VerticalRegion(const std::shared_ptr<sir::AST>& ast,
                 const std::shared_ptr<Interval>& verticalInterval, LoopOrderKind loopOrder,
                 std::optional<Interval> iterationSpaceI, std::optional<Interval> iterationSpaceJ,
                 SourceLocation loc = SourceLocation())
      : Loc(loc), Ast(ast), VerticalInterval(verticalInterval), LoopOrder(loopOrder),
        IterationSpace({iterationSpaceI, iterationSpaceJ}) {}

  /// @brief Clone the vertical region
  std::shared_ptr<VerticalRegion> clone() const;

  /// @brief Comparison between stencils (omitting location)
  bool operator==(const VerticalRegion& rhs) const;

  /// @brief Comparison between stencils (omitting location)
  /// if the comparison fails, outputs human readable reason why in the string
  CompareResult comparison(const VerticalRegion& rhs) const;
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
  std::shared_ptr<sir::AST> StencilDescAst;   ///< Stencil description AST
  std::vector<std::shared_ptr<Field>> Fields; ///< Fields referenced by this stencil
  Attr Attributes;                            ///< Attributes of the stencil

  bool operator==(const Stencil& rhs) const;
  CompareResult comparison(const Stencil& rhs) const;
};

//===------------------------------------------------------------------------------------------===//
//     Global Variable Map
//===------------------------------------------------------------------------------------------===//

/// @brief Representation of a value of a global variable
///
/// This is a very primitive (i.e non-copyable, immutable) version of boost::any.
/// Further, the possible values are restricted to `bool`, `int`, `double` or `std::string`.
///
/// @ingroup sir

class Value {
public:
  enum class Kind { Boolean = 0, Integer, Float, Double, String };
  template <typename T>
  struct TypeInfo;

  template <class T>
  Value(T value)
      : value_{std::move(value)}, is_constexpr_{false}, type_{TypeInfo<std::decay_t<T>>::Type} {}

  template <class T>
  Value(T value, bool is_constexpr)
      : value_{std::move(value)},
        is_constexpr_{is_constexpr}, type_{TypeInfo<std::decay_t<T>>::Type} {}

  Value(const Value& other)
      : value_{other.value_}, is_constexpr_{other.isConstexpr()}, type_{other.getType()} {}

  Value(Value& other)
      : value_{other.value_}, is_constexpr_{other.isConstexpr()}, type_{other.getType()} {}

  Value(Value&& other)
      : value_{std::move(other.value_)}, is_constexpr_{other.isConstexpr()}, type_{
                                                                                 other.getType()} {}

  Value(Kind type) : value_{}, is_constexpr_{false}, type_{type} {}

  virtual ~Value() = default;

  Value& operator=(const Value& other) {
    value_ = other.value_;
    is_constexpr_ = other.isConstexpr();
    type_ = other.getType();
    return *this;
  }

  /// @brief Get/Set if the variable is `constexpr`
  virtual bool isConstexpr() const { return is_constexpr_; }

  /// @brief `Type` to string
  static const char* typeToString(Kind type);

  /// @brief `Type` to `BuiltinTypeID`
  static BuiltinTypeID typeToBuiltinTypeID(Kind type);

  /// Convert the value to string
  virtual std::string toString() const;

  /// @brief Check if value is set
  virtual bool has_value() const { return value_.has_value(); }

  /// @brief Get/Set the underlying type
  virtual Kind getType() const { return type_; }

  /// @brief Get the value as type `T`
  /// @returns Copy of the value
  template <class T>
  T getValue() const {
    DAWN_ASSERT(has_value());
    DAWN_ASSERT_MSG(getType() == TypeInfo<T>::Type, "type mismatch");
    return std::get<T>(*value_);
  }

  virtual bool operator==(const Value& rhs) const;
  virtual CompareResult comparison(const sir::Value& rhs) const;

  virtual json::json jsonDump() const {
    json::json valueJson;
    valueJson["type"] = Value::typeToString(getType());
    valueJson["isConstexpr"] = isConstexpr();
    valueJson["value"] = toString();
    return valueJson;
  }

protected:
  std::optional<std::variant<bool, int, float, double, std::string>> value_;
  bool is_constexpr_;
  Kind type_;
};

template <>
struct Value::TypeInfo<bool> {
  static constexpr Kind Type = Kind::Boolean;
};

template <>
struct Value::TypeInfo<int> {
  static constexpr Kind Type = Kind::Integer;
};

template <>
struct Value::TypeInfo<float> {
  static constexpr Kind Type = Kind::Float;
};

template <>
struct Value::TypeInfo<double> {
  static constexpr Kind Type = Kind::Double;
};

template <>
struct Value::TypeInfo<std::string> {
  static constexpr Kind Type = Kind::String;
};

class Global : public Value, NonCopyable {
public:
  template <class T>
  Global(T value) : Value(value) {}

  template <class T>
  Global(T value, bool is_constexpr) : Value(value, is_constexpr) {}

  Global(Kind type) : Value(type) {}
};

using GlobalVariableMap = std::unordered_map<std::string, Global>;

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
  void dump();

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
  std::shared_ptr<sir::GlobalVariableMap> GlobalVariableMap;           ///< Map of global variables
  const ast::GridType GridType;
};

} // namespace dawn

#endif
