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

#include <iosfwd>
#include <string>

namespace dawn {

/// @brief Builtin types recognized by the SIR
/// @ingroup support
enum class BuiltinTypeID : int { Invalid = 0, Auto, Boolean, Integer, Float, Double };

/// @brief Const-volatile qualifiers
/// @ingroup support
enum class CVQualifier : int { Invalid = 0, Const = 1 << 2, Volatile = 1 << 3 };

inline CVQualifier operator|(const CVQualifier& lhs, const CVQualifier& rhs) {
  return static_cast<CVQualifier>(static_cast<int>(lhs) | static_cast<int>(rhs));
}

inline CVQualifier operator|=(CVQualifier& lhs, const CVQualifier& rhs) {
  return (lhs = static_cast<CVQualifier>(static_cast<int>(lhs) | static_cast<int>(rhs)));
}

/// @brief Type representation
/// @ingroup support
class Type {
  std::string name_;
  BuiltinTypeID builtinTypeID_;
  CVQualifier cvQualifier_;

public:
  template <typename T>
  struct TypeInfo;

  /// @name Constructors and Assignment
  /// @{
  Type() : name_(), builtinTypeID_(BuiltinTypeID::Invalid), cvQualifier_(CVQualifier::Invalid) {}
  Type(BuiltinTypeID builtinTypeID, CVQualifier cvQualifier = CVQualifier::Invalid)
      : name_(), builtinTypeID_(builtinTypeID), cvQualifier_(cvQualifier) {}
  Type(const std::string& name, CVQualifier cvQualifier = CVQualifier::Invalid)
      : name_(name), builtinTypeID_(BuiltinTypeID::Invalid), cvQualifier_(cvQualifier) {}
  Type(const Type&) = default;
  Type(Type&&) = default;

  Type& operator=(const Type&) = default;
  Type& operator=(Type&&) = default;
  /// @}

  const BuiltinTypeID& getBuiltinTypeID() const { return builtinTypeID_; }
  BuiltinTypeID& getBuiltinTypeID() { return builtinTypeID_; }

  const std::string& getName() const { return name_; }
  std::string& getName() { return name_; }

  CVQualifier& getCVQualifier() { return cvQualifier_; }
  const CVQualifier& getCVQualifier() const { return cvQualifier_; }

  bool isBuiltinType() const { return builtinTypeID_ != BuiltinTypeID::Invalid; }
  bool isConst() const {
    return static_cast<int>(cvQualifier_) & static_cast<int>(CVQualifier::Const);
  }
  bool isVolatile() const {
    return static_cast<int>(cvQualifier_) & static_cast<int>(CVQualifier::Volatile);
  }

  /// @name Comparison
  /// @{
  bool operator==(const Type& other) const {
    return name_ == other.name_ && builtinTypeID_ == other.builtinTypeID_ &&
           cvQualifier_ == other.cvQualifier_;
  }
  bool operator!=(const Type& other) const { return !(*this == other); }
  /// @}

  /// @brief Stream to C++ compatible type
  friend std::ostream& operator<<(std::ostream& os, Type type);
};

template <>
struct Type::TypeInfo<bool> {
  static constexpr BuiltinTypeID Type = BuiltinTypeID::Boolean;
};

template <>
struct Type::TypeInfo<int> {
  static constexpr BuiltinTypeID Type = BuiltinTypeID::Integer;
};

template <>
struct Type::TypeInfo<float> {
  static constexpr BuiltinTypeID Type = BuiltinTypeID::Float;
};

template <>
struct Type::TypeInfo<double> {
  static constexpr BuiltinTypeID Type = BuiltinTypeID::Double;
};

extern std::ostream& operator<<(std::ostream& os, Type type);

} // namespace dawn
