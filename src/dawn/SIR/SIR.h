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

#include "dawn/SIR/AST.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/NonCopyable.h"
#include "dawn/Support/SourceLocation.h"
#include "dawn/Support/Type.h"
#include <iosfwd>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace dawn {

/// @namespace sir
/// @brief This namespace contains a C++ implementation of the SIR specification
/// @ingroup sir
namespace sir {

/// @brief Result of comparisons
/// contains the boolean that is true when the comparee match and an error message if not
struct CompareResult {
  std::string message;
  bool match;

  operator bool() { return match; }
  std::string why() { return message; }
};

/// @brief Attributes attached to various SIR objects which allow to change the behavior on per
/// stencil basis
/// @ingroup sir
class Attr {
  unsigned attrBits_;

public:
  Attr() : attrBits_(0) {}

  /// @brief Attribute bit-mask
  enum AttrKind : unsigned {
    AK_NoCodeGen = 1 << 0,        ///< Don't generate code for this stencil
    AK_MergeStages = 1 << 1,      ///< Merge the Stages of this stencil
    AK_MergeDoMethods = 1 << 2,   ///< Merge the Do-Methods of this stencil
    AK_MergeTemporaries = 1 << 3, ///< Merge the temporaries of this stencil
    AK_UseKCaches = 1 << 4        ///< Use K-Caches
  };

  /// @brief Check if `attr` bit is set
  bool has(AttrKind attr) const { return (attrBits_ >> attr) & 1; }

  /// @brief Check if any of the `attrs` bits is set
  /// @{
  bool hasOneOf(AttrKind attr1, AttrKind attr2) const { return has(attr1) || has(attr2); }

  template <typename... AttrTypes>
  bool hasOneOf(AttrKind attr1, AttrKind attr2, AttrTypes... attrs) const {
    return has(attr1) || hasOneOf(attr2, attrs...);
  }
  /// @}

  ///@brief getting the Bits
  unsigned getBits() const { return attrBits_; }
  /// @brief Set `attr`bit
  void set(AttrKind attr) { attrBits_ |= 1 << attr; }

  /// @brief Unset `attr` bit
  void unset(AttrKind attr) { attrBits_ &= ~(1 << attr); }

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
  enum ArgumentKind { AK_Field, AK_Direction, AK_Offset };

  static constexpr const int AK_NumArgTypes = 3;

  std::string Name;   ///< Name of the argument
  ArgumentKind Kind;  ///< Type of argument
  SourceLocation Loc; ///< Source location

  bool operator==(const StencilFunctionArg& rhs) const;
  CompareResult comparison(const sir::StencilFunctionArg& rhs) const;
};

/// @brief Representation of a field
/// @ingroup sir
struct Field : public StencilFunctionArg {
  Field(const std::string& name, SourceLocation loc = SourceLocation())
      : StencilFunctionArg{name, AK_Field, loc}, IsTemporary(false), fieldDimensions({{0, 0, 0}}) {}

  bool IsTemporary;
  Array3i fieldDimensions;

  static bool classof(const StencilFunctionArg* arg) { return arg->Kind == AK_Field; }
  bool operator==(const Field& rhs) const { return comparison(rhs); }

  CompareResult comparison(const Field& rhs) const;
};

/// @brief Representation of a direction (e.g `i`)
/// @ingroup sir
struct Direction : public StencilFunctionArg {
  Direction(const std::string& name, SourceLocation loc = SourceLocation())
      : StencilFunctionArg{name, AK_Direction, loc} {}

  static bool classof(const StencilFunctionArg* arg) { return arg->Kind == AK_Direction; }
};

/// @brief Representation of an Offset (e.g `i + 1`)
/// @ingroup sir
struct Offset : public StencilFunctionArg {
  Offset(const std::string& name, SourceLocation loc = SourceLocation())
      : StencilFunctionArg{name, AK_Offset, loc} {}

  static bool classof(const StencilFunctionArg* arg) { return arg->Kind == AK_Offset; }
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
  std::vector<std::shared_ptr<AST>> Asts;           ///< ASTs of the specializations
  Attr Attributes;                                  ///< Attributes of the stencil function

  /// @brief Check if the Stencil function contains specializations
  ///
  /// If `Intervals` is empty and `Asts` contains one element, the StencilFunction is not
  /// specialized.
  bool isSpecialized() const;

  /// @brief Get the AST of the specified vertical interval or `NULL` if the function is not
  /// specialized for this interval
  std::shared_ptr<AST> getASTOfInterval(const Interval& interval) const;

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
  enum LoopOrderKind { LK_Forward = 0, LK_Backward };

  SourceLocation Loc;                         ///< Source location of the vertical region
  std::shared_ptr<AST> Ast;                   ///< AST of the region
  std::shared_ptr<Interval> VerticalInterval; ///< Interval description of the region
  LoopOrderKind LoopOrder;                    /// Loop order (usually associated with the k-loop)

  VerticalRegion(const std::shared_ptr<AST>& ast, const std::shared_ptr<Interval>& verticalInterval,
                 LoopOrderKind loopOrder, SourceLocation loc = SourceLocation())
      : Loc(loc), Ast(ast), VerticalInterval(verticalInterval), LoopOrder(loopOrder) {}

  /// @brief Clone the vertical region
  std::shared_ptr<VerticalRegion> clone() const;

  /// @brief Comparison between stencils (omitting location)
  bool operator==(const VerticalRegion& rhs) const;

  /// @brief Comparison between stencils (omitting location)
  /// if the comparison fails, outputs human readable reason why in the string
  CompareResult comparison(const VerticalRegion& rhs) const;
};

/// @brief Call to another stencil
/// @ingroup sir
struct StencilCall {

  SourceLocation Loc;                       ///< Source location of the call
  std::string Callee;                       ///< Name of the callee stencil
  std::vector<std::shared_ptr<Field>> Args; ///< List of fields used as arguments

  StencilCall(std::string callee, SourceLocation loc = SourceLocation())
      : Loc(loc), Callee(callee) {}

  /// @brief Clone the vertical region
  std::shared_ptr<StencilCall> clone() const;
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
  std::shared_ptr<AST> StencilDescAst;        ///< Stencil description AST
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
/// This is a very primitive (i.e non-copyable) version of boost::any. Further, the possible
/// values are restricted to `bool`, `int`, `double` or `std::string`.
///
/// @ingroup sir
struct Value : NonCopyable {
  enum TypeKind { None = 0, Boolean, Integer, Double, String };

  Value() : type_(None), isConstexpr_(false), valueImpl_(nullptr) {}

  template <class T>
  explicit Value(T&& value) : isConstexpr_(false) {
    setValue(value);
  }

  /// @brief Get/Set if the variable is `constexpr`
  bool isConstexpr() const { return isConstexpr_; }
  void setIsConstexpr(bool isConstexpr) { isConstexpr_ = isConstexpr; }

  /// @brief `TypeKind` to string
  static const char* typeToString(TypeKind type);

  /// @brief `TypeKind` to `BuiltinTypeID`
  static BuiltinTypeID typeToBuiltinTypeID(TypeKind type);

  /// Convert the value to string
  std::string toString() const;

  /// @brief Check if value is empty
  bool empty() const { return valueImpl_ == nullptr; }

  /// @brief Get/Set the underlying type
  TypeKind getType() const { return type_; }
  void setType(TypeKind type) { type_ = type; }

  /// @brief Get the value as type `T`
  /// @returns Copy of the value
  template <class T>
  T getValue() const {
    DAWN_ASSERT(!empty());
    DAWN_ASSERT_MSG(TypeInfo<T>::Type == type_, "type mismatch");
    return *(T*)valueImpl_->get();
  }

  /// @brief Set the value
  template <class T>
  void setValue(const T& value) {
    valueImpl_ = make_unique<ValueImpl<T>>(value);
    type_ = TypeInfo<T>::Type;
  }

  struct EmptyType {};
  template <class T>
  struct TypeInfo {
    static const TypeKind Type = None;
  };

  bool operator==(const Value& rhs) const;
  CompareResult comparison(const sir::Value& rhs) const;

private:
  struct ValueImplBase {
    virtual ~ValueImplBase() {}
    virtual void* get() = 0;
  };

  template <class T>
  struct ValueImpl : public ValueImplBase {
    T* ValuePtr;

    ValueImpl() : ValuePtr(nullptr) {}
    ValueImpl(const T& value) : ValuePtr(new T) { *ValuePtr = value; }
    ~ValueImpl() {
      if(ValuePtr)
        delete ValuePtr;
    }
    void* get() override { return (void*)ValuePtr; }
  };

  TypeKind type_;
  bool isConstexpr_;
  std::unique_ptr<ValueImplBase> valueImpl_;
};

template <>
struct Value::TypeInfo<bool> {
  static const Value::TypeKind Type = Value::Boolean;
};

template <>
struct Value::TypeInfo<int> {
  static const Value::TypeKind Type = Value::Integer;
};

template <>
struct Value::TypeInfo<double> {
  static const Value::TypeKind Type = Value::Double;
};

template <>
struct Value::TypeInfo<std::string> {
  static const Value::TypeKind Type = Value::String;
};

/// @brief Representation of the global variable map (key/value pair)
/// @ingroup sir
using GlobalVariableMap = std::unordered_map<std::string, std::shared_ptr<sir::Value>>;

} // namespace sir

//===------------------------------------------------------------------------------------------===//
//     SIR
//===------------------------------------------------------------------------------------------===//

/// @brief Definition of the Stencil Intermediate Representation (SIR)
/// @ingroup sir
struct SIR : public dawn::NonCopyable {

  /// @brief Default Ctor that initializes all the shared pointers
  SIR();

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
  sir::CompareResult comparison(const SIR& rhs) const;

  /// @brief Dump SIR to the given stream
  friend std::ostream& operator<<(std::ostream& os, const SIR& Sir);

  std::string Filename;                                ///< Name of the file the SIR was parsed from
  std::vector<std::shared_ptr<sir::Stencil>> Stencils; ///< List of stencils
  std::vector<std::shared_ptr<sir::StencilFunction>> StencilFunctions; ///< List of stencil function
  std::shared_ptr<sir::GlobalVariableMap> GlobalVariableMap;           ///< Map of global variables
};

} // namespace dawn

#endif
