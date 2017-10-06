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

#include "dawn/SIR/SIR.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/Printing.h"
#include "dawn/Support/StringUtil.h"
#include "dawn/Support/Unreachable.h"
#include <iostream>
#include <sstream>

namespace dawn {

bool sir::StencilFunction::isSpecialized() const { return !Intervals.empty(); }

std::shared_ptr<AST> sir::StencilFunction::getASTOfInterval(const sir::Interval& interval) const {
  for(int i = 0; i < Intervals.size(); ++i)
    if(*Intervals[i] == interval)
      return Asts[i];
  return nullptr;
}

std::string sir::Interval::toString() const {
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

namespace sir {

std::ostream& operator<<(std::ostream& os, const Interval& interval) {
  auto printLevel = [&os](int level, int offset) -> void {
    if(level == sir::Interval::Start)
      os << "Start";
    else if(level == sir::Interval::End)
      os << "End";
    else
      os << level;

    if(offset != 0)
      os << (offset > 0 ? "+" : "") << offset;
  };

  os << "{ ";
  printLevel(interval.LowerLevel, interval.LowerOffset);
  os << " : ";
  printLevel(interval.UpperLevel, interval.UpperOffset);
  os << " }";
  return os;
}

} // namespace sir

std::ostream& operator<<(std::ostream& os, const SIR& Sir) {
  const char* indent1 = MakeIndent<1>::value;
  const char* indent2 = MakeIndent<2>::value;

  os << "SIR : " << Sir.Filename << "\n{\n";
  for(const auto& stencilFunction : Sir.StencilFunctions) {
    os << "\n"
       << indent1 << "StencilFunction : " << stencilFunction->Name << "\n"
       << indent1 << "{\n";
    for(const auto& arg : stencilFunction->Args) {
      if(sir::Field* field = dyn_cast<sir::Field>(arg.get()))
        os << indent2 << "Field : " << field->Name << "\n";
      if(sir::Direction* dir = dyn_cast<sir::Direction>(arg.get()))
        os << indent2 << "Direction : " << dir->Name << "\n";
      if(sir::Offset* offset = dyn_cast<sir::Offset>(arg.get()))
        os << indent2 << "Offset : " << offset->Name << "\n";
    }

    if(!stencilFunction->isSpecialized()) {
      os << "\n" << indent2 << "Do\n";
      os << ASTStringifer::toString(*stencilFunction->Asts[0], 2 * DAWN_PRINT_INDENT);
    } else {
      for(int i = 0; i < stencilFunction->Intervals.size(); ++i) {
        os << "\n" << indent2 << "Do " << *stencilFunction->Intervals[i].get() << "\n";
        os << ASTStringifer::toString(*stencilFunction->Asts[i], 2 * DAWN_PRINT_INDENT);
      }
    }
    os << indent1 << "}\n";
  }

  for(const auto& stencil : Sir.Stencils) {
    os << "\n" << indent1 << "Stencil : " << stencil->Name << "\n" << indent1 << "{\n";
    for(const auto& field : stencil->Fields)
      os << indent2 << "Field : " << field->Name << "\n";
    os << "\n";

    os << indent2 << "Do\n"
       << ASTStringifer::toString(*stencil->StencilDescAst, 2 * DAWN_PRINT_INDENT);
    os << indent1 << "}\n";
  }

  os << "\n}";
  return os;
}

void SIR::dump() { std::cout << *this << std::endl; }

const char* sir::Value::typeToString(sir::Value::TypeKind type) {
  switch(type) {
  case None:
    return "None";
  case Boolean:
    return "bool";
  case Integer:
    return "int";
  case Double:
    return "double";
  case String:
    return "std::string";
  }
  dawn_unreachable("invalid type");
}

BuiltinTypeID sir::Value::typeToBuiltinTypeID(sir::Value::TypeKind type) {
  switch(type) {
  case None:
    return BuiltinTypeID::None;
  case Boolean:
    return BuiltinTypeID::Boolean;
  case Integer:
    return BuiltinTypeID::Integer;
  case Double:
    return BuiltinTypeID::Float;
  default:
    dawn_unreachable("invalid type");
  }
}

std::string sir::Value::toString() const {
  if(empty())
    return "null";

  std::stringstream ss;
  switch(type_) {
  case Boolean:
    ss << (getValue<bool>() ? "true" : "false");
    break;
  case Integer:
    ss << getValue<int>();
    break;
  case Double:
    ss << getValue<double>();
    break;
  case String:
    ss << "\"" << getValue<std::string>() << "\"";
    break;
  default:
    dawn_unreachable("invalid type");
  }
  return ss.str();
}

std::shared_ptr<sir::VerticalRegion> sir::VerticalRegion::clone() const {
  return std::make_shared<sir::VerticalRegion>(Ast->clone(), VerticalInterval, LoopOrder, Loc);
}

std::shared_ptr<sir::StencilCall> sir::StencilCall::clone() const {
  auto call = std::make_shared<sir::StencilCall>(Callee, Loc);
  call->Args = Args;
  return call;
}

bool SIR::operator==(const SIR& rhs) const {
  bool retval = true;

  if((Stencils.size() == rhs.Stencils.size()) &&
     (StencilFunctions.size() == rhs.StencilFunctions.size())) {

    if(Stencils.size() > 0) {
      retval &= std::equal(Stencils.begin(), Stencils.end(), rhs.Stencils.begin(),
                           pointeeComparison<sir::Stencil>);
    }

    if(StencilFunctions.size() > 0) {
      retval &=
          std::equal(StencilFunctions.begin(), StencilFunctions.end(), rhs.StencilFunctions.begin(),
                     pointeeComparison<sir::StencilFunction>);
    }

    if(GlobalVariableMap != nullptr && rhs.GlobalVariableMap != nullptr) {
      if(GlobalVariableMap->size() == rhs.GlobalVariableMap->size()) {
        if(GlobalVariableMap->size() > 0) {
          retval &=
              pointerMapComparison(*(GlobalVariableMap.get()), *(rhs.GlobalVariableMap.get()));
        }
      } else {
        return false;
      }
    } else {
      return GlobalVariableMap == rhs.GlobalVariableMap;
    }

    return retval;
  }
  return false;
}
bool SIR::operator!=(const SIR& rhs) const { return !(*this == rhs); }

bool sir::Stencil::operator==(const sir::Stencil& rhs) const {
  bool retval = true;
  if(Fields.size() != rhs.Fields.size()) {
    return false;
  }
  retval &= (Name == rhs.Name);
  retval &= (Attributes == rhs.Attributes);
  if(StencilDescAst != nullptr && rhs.StencilDescAst != nullptr) {
    if(StencilDescAst->getRoot() != nullptr && rhs.StencilDescAst->getRoot() != nullptr) {
      retval &= StencilDescAst->getRoot().get()->equals(rhs.StencilDescAst->getRoot().get());
    } else {
      return StencilDescAst->getRoot() == rhs.StencilDescAst->getRoot();
    }
  } else {
    return StencilDescAst == rhs.StencilDescAst;
  }
  if(Fields.size() > 0) {
    retval &= std::equal(Fields.begin(), Fields.end(), rhs.Fields.begin(),
                         pointeeComparison<Field>);
  }

  return retval;
}

bool sir::StencilFunction::operator==(const sir::StencilFunction& rhs) const {
  bool retval = true;
  retval &= (Name == rhs.Name);
  retval &= Attributes == rhs.Attributes;
  if(Args.size() != rhs.Args.size() && Intervals.size() != rhs.Intervals.size() &&
     Asts.size() != rhs.Asts.size()) {
    return false;
  } else {
    if(Args.size() > 0) {
      retval &= std::equal(Args.begin(), Args.end(), rhs.Args.begin(),
                           pointeeComparison<sir::StencilFunctionArg>);
    }
    if(Intervals.size() > 0) {
      retval &= std::equal(Intervals.begin(), Intervals.end(), rhs.Intervals.begin(),
                           pointeeComparison<sir::Interval>);
    }

    if(Asts.size() > 0) {
      auto astcomparison = [](std::shared_ptr<dawn::AST> comparate1,
                              std::shared_ptr<dawn::AST> comparate2) {
        if(comparate1->getRoot() != nullptr && comparate2->getRoot() != nullptr) {
          return comparate1->getRoot().get()->equals(comparate2->getRoot().get());
        } else {
          return comparate1->getRoot() == comparate2->getRoot();
        }
      };
      retval &= std::equal(Asts.begin(), Asts.end(), rhs.Asts.begin(), astcomparison);
    }
  }
  return retval;
}

bool sir::StencilFunctionArg::operator==(const sir::StencilFunctionArg& rhs) const {
  return (Name == rhs.Name) && (Kind == rhs.Kind);
}
bool SIR::pointerMapComparison(const sir::GlobalVariableMap& map1,
                               const sir::GlobalVariableMap& map2) const {
  if(map1.size() != map2.size()) {
    return false;
  } else {
    for(auto& a : map1) {
      auto finder = map2.find(a.first);
      if(finder == map2.end()) {
        return false;
      } else if(finder->second == nullptr || a.second == nullptr) {
        return finder->second == a.second;
      } else if(!(*(finder->second.get()) == *(a.second.get()))) {
        return false;
      }
    }
    return true;
  }
}

bool sir::Value::operator==(const sir::Value& rhs) const {
  if(getType() != rhs.getType()) {
    return false;
  }
  if(getType() == Value::TypeKind::Boolean) {
    return getValue<bool>() == rhs.getValue<bool>();
  }
  if(getType() == Value::TypeKind::Integer) {
    return getValue<int>() == rhs.getValue<int>();
  }
  if(getType() == Value::TypeKind::Double) {
    return getValue<double>() == rhs.getValue<double>();
  }
  if(getType() == Value::TypeKind::String) {
    return getValue<std::string>() == rhs.getValue<std::string>();
  }
  return true;
}

bool pointeeComparison(const std::shared_ptr<T>& comparate1,
                              const std::shared_ptr<T>& comparate2) {
  return *comparate1 == *comparate2;
}

} // namespace dawn
