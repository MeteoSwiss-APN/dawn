//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gsl/Support/Type.h"
#include "gsl/Support/Unreachable.h"
#include <iostream>

namespace gsl {

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
      gsl_unreachable("invalid BuiltinTypeID");
    }
  else {
    os << type.name_;
  }

  return os;
}

} // namespace gsl
