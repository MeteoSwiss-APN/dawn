#include "Value.h"
#include "dawn/Support/Unreachable.h"
#include <iomanip>

namespace dawn {
namespace ast {

CompareResult Value::comparison(const Value& rhs) const {
  auto type = getType();
  if(type != rhs.getType())
    return CompareResult{dawn::format("[Value mismatch] Values are not of the same type\n"
                                      "  Actual:\n"
                                      "    %s\n"
                                      "  Expected:\n"
                                      "    %s",
                                      Value::typeToString(type),
                                      Value::typeToString(rhs.getType())),
                         false};

  switch(type) {
  case Value::Kind::Boolean:
    return isEqualImpl<bool>(*this, rhs, rhs.toString());
  case Value::Kind::Integer:
    return isEqualImpl<int>(*this, rhs, rhs.toString());
  case Value::Kind::Double:
    return isEqualImpl<double>(*this, rhs, rhs.toString());
  case Value::Kind::Float:
    return isEqualImpl<float>(*this, rhs, rhs.toString());
  case Value::Kind::String:
    return isEqualImpl<std::string>(*this, rhs, rhs.toString());
  default:
    dawn_unreachable("invalid type");
  }
}

const char* Value::typeToString(Value::Kind type) {
  switch(type) {
  case Kind::Boolean:
    return "bool";
  case Kind::Integer:
    return "int";
  case Kind::Double:
    return "double";
  case Kind::Float:
    return "float";
  case Kind::String:
    return "std::string";
  }
  dawn_unreachable("invalid type");
}

BuiltinTypeID Value::typeToBuiltinTypeID(Value::Kind type) {
  switch(type) {
  case Kind::Boolean:
    return BuiltinTypeID::Boolean;
  case Kind::Integer:
    return BuiltinTypeID::Integer;
  case Kind::Double:
    return BuiltinTypeID::Double;
  case Kind::Float:
    return BuiltinTypeID::Float;
  default:
    dawn_unreachable("invalid type");
  }
}

std::string Value::toString() const {
  std::ostringstream out;
  DAWN_ASSERT(has_value());
  switch(type_) {
  case Kind::Boolean:
    return std::get<bool>(*value_) ? "true" : "false";
  case Kind::Integer:
    return std::to_string(std::get<int>(*value_));
  case Kind::Double:
    out << std::setprecision(std::numeric_limits<double>::max_digits10)
        << std::get<double>(*value_);
    return out.str();
  case Kind::Float:
    out << std::setprecision(std::numeric_limits<float>::max_digits10) << std::get<float>(*value_);
    return out.str();
  case Kind::String:
    return std::get<std::string>(*value_);
  default:
    dawn_unreachable("invalid type");
  }
}


bool Value::operator==(const Value& rhs) const { return bool(comparison(rhs)); }

} // namespace sir
} // namespace dawn
