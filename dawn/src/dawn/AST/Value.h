#pragma once

#include "dawn/Support/ComparisonHelpers.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Json.h"
#include "dawn/Support/NonCopyable.h"
#include "dawn/Support/Type.h"
#include <variant>

namespace dawn {
namespace ast {

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
      : value_{std::move(value)}, isConstexpr_{false}, type_{TypeInfo<std::decay_t<T>>::Type} {}

  template <class T>
  Value(T value, bool is_constexpr)
      : value_{std::move(value)},
        isConstexpr_{is_constexpr}, type_{TypeInfo<std::decay_t<T>>::Type} {}

  Value(const Value& other)
      : value_{other.value_}, isConstexpr_{other.isConstexpr()}, type_{other.getType()} {}

  Value(Value& other)
      : value_{other.value_}, isConstexpr_{other.isConstexpr()}, type_{other.getType()} {}

  Value(Value&& other)
      : value_{std::move(other.value_)}, isConstexpr_{other.isConstexpr()}, type_{other.getType()} {
  }

  Value(Kind type) : value_{}, isConstexpr_{false}, type_{type} {}

  virtual ~Value() = default;

  Value& operator=(const Value& other) {
    value_ = other.value_;
    isConstexpr_ = other.isConstexpr();
    type_ = other.getType();
    return *this;
  }

  /// @brief Get/Set if the variable is `constexpr`
  virtual bool isConstexpr() const { return isConstexpr_; }

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
  virtual CompareResult comparison(const Value& rhs) const;

  virtual json::json jsonDump() const {
    json::json valueJson;
    valueJson["type"] = Value::typeToString(getType());
    valueJson["isConstexpr"] = isConstexpr();
    valueJson["value"] = toString();
    return valueJson;
  }

protected:
  std::optional<std::variant<bool, int, float, double, std::string>> value_;
  bool isConstexpr_;
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

// Using ordered map to guarantee the same backend code will be generated
using GlobalVariableMap = std::map<std::string, Global>;

} // namespace sir

namespace {

///@brief Stringification of a Value mismatch
template <class T>
CompareResult isEqualImpl(const ast::Value& a, const ast::Value& b, const std::string& name) {
  if(a.getValue<T>() != b.getValue<T>())
    return CompareResult{dawn::format("[Value mismatch] %s values are not equal\n"
                                      "  Actual:\n"
                                      "    %s\n"
                                      "  Expected:\n"
                                      "    %s",
                                      name, a.toString(), b.toString()),
                         false};

  return CompareResult{"", true};
}

}

} // namespace dawn
