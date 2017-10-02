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

#include "dawn/Support/Type.h"
#include "dawn/Support/Unreachable.h"
#include <iostream>

namespace dawn {

std::ostream& operator<<(std::ostream& os, Type type) {
  if(type.isConst())
    os << "const ";
  if(type.isVolatile())
    os << "volatile ";

  if(type.isBuiltinType())
    switch(type.builtinTypeID_) {
    case BuiltinTypeID::Auto:
      os << "auto";
      break;
    case BuiltinTypeID::Boolean:
      os << "bool";
      break;
    case BuiltinTypeID::Integer:
      os << "int_type";
      break;
    case BuiltinTypeID::Float:
      os << "float_type";
      break;
    default:
      dawn_unreachable("invalid BuiltinTypeID");
    }
  else {
    os << type.name_;
  }

  return os;
}

} // namespace dawn
