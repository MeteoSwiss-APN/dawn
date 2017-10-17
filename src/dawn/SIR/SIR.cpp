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
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Printing.h"
#include "dawn/Support/StringUtil.h"
#include "dawn/Support/Unreachable.h"
#include <iostream>
#include <sstream>

namespace dawn {

namespace {

/// @brief Allow direct comparison of the Stmts of an AST
class DiffWriter final : public ASTVisitorForwarding {
public:
  virtual void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) override {
    statements_.push_back(stmt);
    stmt->getVerticalRegion()->Ast->getRoot()->accept(*this);
  }

  virtual void visit(const std::shared_ptr<ReturnStmt>& stmt) override {
    statements_.push_back(stmt);
    ASTVisitorForwarding::visit(stmt);
  }

  virtual void visit(const std::shared_ptr<ExprStmt>& stmt) override {
    statements_.push_back(stmt);
    ASTVisitorForwarding::visit(stmt);
  }

  virtual void visit(const std::shared_ptr<BlockStmt>& stmt) override {
    statements_.push_back(stmt);
    ASTVisitorForwarding::visit(stmt);
  }

  virtual void visit(const std::shared_ptr<VarDeclStmt>& stmt) override {
    statements_.push_back(stmt);
    ASTVisitorForwarding::visit(stmt);
  }

  std::vector<std::shared_ptr<Stmt>> getStatements() const { return statements_; }

  std::pair<std::string, bool> compare(const DiffWriter& other) {
    std::size_t minSize = std::min(statements_.size(), other.getStatements().size());
    if(minSize == 0 && (statements_.size() != other.getStatements().size()))
      return std::make_pair("[AST mismatch] AST is empty", false);

    for(std::size_t idx = 0; idx < minSize; ++idx) {
      if(!statements_[idx]->equals(other.getStatements()[idx].get())) {
        return std::make_pair(
            dawn::format("[AST mismatch] Statement mismatch\n"
                         "  Actual:\n"
                         "    %s\n"
                         "  Expected:\n"
                         "    %s",
                         indent(ASTStringifer::toString(statements_[idx]), 4),
                         indent(ASTStringifer::toString(other.getStatements()[idx]), 4)),
            false);
      }
    }

    return std::make_pair("", true);
  }

private:
  std::vector<std::shared_ptr<Stmt>> statements_;
};

///@brief Stringification of a Value mismatch
template <class T>
std::pair<std::string, bool> isEqualImpl(const sir::Value& a, const sir::Value& b,
                                         const std::string& name) {
  if(a.getValue<T>() != b.getValue<T>())
    return std::make_pair(dawn::format("[Value mismatch] %s values are not equal\n"
                                       "  Actual:\n"
                                       "    %s\n"
                                       "  Expected:\n"
                                       "    %s",
                                       name, a.toString(), b.toString()),
                          false);

  return std::make_pair("", true);
}

/// @brief Compares two ASTs
std::pair<std::string, bool> compareAst(const std::shared_ptr<AST>& lhs,
                                        const std::shared_ptr<AST>& rhs) {
  if(lhs->getRoot()->equals(rhs->getRoot().get()))
    return std::make_pair("", true);

  DiffWriter lhsDW, rhsDW;
  lhs->accept(lhsDW);
  rhs->accept(rhsDW);

  auto comp = lhsDW.compare(rhsDW);
  if(!comp.second)
    return comp;

  return std::make_pair("", true);
}

/// @brief Compares the content of two shared pointers
///
/// @param[in] shared pointer of type T
/// @param[in] shared pointer of same type T
/// @return true if contents of the shared pointers match (operator ==)
template <typename T>
bool pointeeComparison(const std::shared_ptr<T>& comparate1, const std::shared_ptr<T>& comparate2) {
  return *comparate1 == *comparate2;
}

/// @brief Compares the content of two shared pointers
///
/// The boolean is true if contents of the shared pointers match (operator ==) the string returns a
/// potential mismatch notification
///
/// @param[in] shared pointer of type T
/// @param[in] shared pointer of same type T
/// @return pair of boolean and string
/// @pre Type T requies a comparison function that returns the pair of bool and string
template <typename T>
std::pair<std::string, bool> pointeeComparisonWithOutput(const std::shared_ptr<T>& comparate1,
                                                         const std::shared_ptr<T>& comparate2) {
  return (*comparate1).comparison(*comparate2);
}

/// @brief Helperfunction to compare two maps of key and shared pointer
/// @return pair of boolean and string
/// the boolean is true if contents of the shared pointers match for every key (operator ==)
/// the string returns a potential mismatch notification
static std::pair<std::string, bool> pointerMapComparison(const sir::GlobalVariableMap& map1,
                                                         const sir::GlobalVariableMap& map2) {
  std::string output;
  if(map1.size() != map2.size()) {
    output += dawn::format("[VariableMap mismatch] Number of Global Varialbes do not match\n"
                           "Expected %i and received %i",
                           map1.size(), map2.size());
    return std::make_pair(output, false);

  } else {
    for(auto& a : map1) {
      auto finder = map2.find(a.first);
      if(finder == map2.end()) {
        output += dawn::format("[VariableMap mismatch] Could not find global variable\n"
                               "Global Variable %s was expected but not found",
                               a.first);
        return std::make_pair(output, false);
      } else if(!(*(finder->second.get()) == *(a.second.get()))) {
        output += dawn::format("[VariableMap mismatch] Global Variables have different values\n"
                               "Global Variable %s has values %i and %i",
                               a.first, a.second->toString(), finder->second->toString());
        return std::make_pair(output, false);
      }
    }
    return std::make_pair(output, true);
  }
}

} // anonymous namespace

std::pair<std::string, bool> SIR::comparison(const SIR& rhs) const {
  std::string output;
  if((Stencils.size() != rhs.Stencils.size()))
    return std::make_pair("[SIR mismatch] number of stencils do not match\n", false);
  if(StencilFunctions.size() != rhs.StencilFunctions.size())
    return std::make_pair("[SIR mismatch] number of stencil functions does not match\n", false);
  if(GlobalVariableMap.get()->size() != rhs.GlobalVariableMap.get()->size())
    return std::make_pair("[SIR mismatch] number of global variables does not match\n", false);

  if(!Stencils.empty() &&
     !std::equal(Stencils.begin(), Stencils.end(), rhs.Stencils.begin(),
                 pointeeComparison<sir::Stencil>)) {
    output += "[SIR mismatch] stencils do not match\n";
    for(unsigned i = 0; i < Stencils.size(); ++i) {
      auto tmp = pointeeComparisonWithOutput(Stencils[i], rhs.Stencils[i]);
      if(tmp.second == false) {
        output += tmp.first;
      }
    }
    return std::make_pair(output, false);
  }
  if(!StencilFunctions.empty() &&
     !std::equal(StencilFunctions.begin(), StencilFunctions.end(), rhs.StencilFunctions.begin(),
                 pointeeComparison<sir::StencilFunction>)) {
    output += "[SIR mismatch] stencil functions do not match\n";
    for(unsigned i = 0; i < StencilFunctions.size(); ++i) {
      auto tmp = pointeeComparisonWithOutput(StencilFunctions[i], rhs.StencilFunctions[i]);
      if(tmp.second == false) {
        output += "mismatch of function " + StencilFunctions[i]->Name + "\n" + tmp.first;
      }
    }
    return std::make_pair(output, false);
  }
  if(!GlobalVariableMap.get()->empty() &&
     !pointerMapComparison(*(GlobalVariableMap.get()), *(rhs.GlobalVariableMap.get())).second) {
    auto a = pointerMapComparison(*(GlobalVariableMap.get()), *(rhs.GlobalVariableMap.get()));
    return std::make_pair(a.first, false);
  }
  return std::make_pair("", true);
}

std::pair<std::string, bool> sir::Stencil::comparison(const sir::Stencil& rhs) const {
  std::string output;
  if(Fields.size() != rhs.Fields.size()) {
    output += dawn::format("[Stencil mismatch] number of Fields does not match\n"
                           "gotten %i and %i Fields\n",
                           Fields.size(), rhs.Fields.size());
    return std::make_pair(output, false);
  }
  if(Name != rhs.Name) {
    output += dawn::format("[Stencil mismatch] Stencil names do not match\n"
                           "Names are \n"
                           "%s\n"
                           "%s\n",
                           Name, rhs.Name);
    return std::make_pair(output, false);
  }
  if(!(Attributes == rhs.Attributes)) {
    output += dawn::format("[Stencil mismatch] Stencil attibutes do not match\n"
                           "Attributes are %i and %i\n",
                           Attributes.getBits(), rhs.Attributes.getBits());
    return std::make_pair(output, false);
  }
  if(!StencilDescAst->getRoot().get()->equals(rhs.StencilDescAst->getRoot().get())) {
    output += "[Stencil mismatch] Stencil ASTs Do not match\n";
    auto a = compareAst(StencilDescAst, rhs.StencilDescAst);
    DAWN_ASSERT_MSG(a.second, "Stencils should not match");
    output += a.first;
    return std::make_pair(output, false);
  }
  if(!Fields.empty() &&
     !std::equal(Fields.begin(), Fields.end(), rhs.Fields.begin(), pointeeComparison<sir::Field>)) {
    output += "[Stencil mismatch] Fields do not match\n";
    for(unsigned i = 0; i < Fields.size(); ++i) {
      auto tmp = pointeeComparisonWithOutput(Fields[i], rhs.Fields[i]);
      if(tmp.second == false) {
        output += "Field " + Fields[i].get()->Name + " mismatch: " + tmp.first;
      }
    }
    return std::make_pair(output, false);
  }

  return std::make_pair(output, true);
}

std::pair<std::string, bool>
sir::StencilFunction::comparison(const sir::StencilFunction& rhs) const {
  std::string output;
  if(Name != rhs.Name) {
    output += dawn::format("[StencilFunction mismatch] Names of Stencil Functions do not match\n"
                           "Found Names %s and %s\n",
                           Name, rhs.Name);
    return std::make_pair(output, false);
  }
  if(!(Attributes == rhs.Attributes)) {
    output +=
        dawn::format("[StencilFunction mismatch] Attributes of Stencil Functions do not match\n"
                     "Found Attributes %i and %i\n",
                     Attributes.getBits(), rhs.Attributes.getBits());
    return std::make_pair(output, false);
  }
  if(Args.size() != rhs.Args.size()) {
    output += dawn::format("[StencilFunction mismatch] Number of Arguments do not match\n"
                           "Found %i and %i arguments respectively\n",
                           Args.size(), rhs.Args.size());
    return std::make_pair(output, false);
  }
  if(Intervals.size() != rhs.Intervals.size()) {
    output += dawn::format("[StencilFunction mismatch] Number of Intervals do not match\n"
                           "Found %i and %i Intervals\n",
                           Intervals.size(), rhs.Intervals.size());
    return std::make_pair(output, false);
  }
  if(Asts.size() != rhs.Asts.size()) {
    output += dawn::format("[StencilFunction mismatch] Number of ASTs does not match\n"
                           "Found %i and %i ASTs",
                           Asts.size(), rhs.Asts.size());
    return std::make_pair(output, false);
  }
  if(!Args.empty() &&
     !std::equal(Args.begin(), Args.end(), rhs.Args.begin(),
                 pointeeComparison<sir::StencilFunctionArg>)) {
    output += "[StencilFunction mismatch] stencil functions arguments do not match\n";
    for(unsigned i = 0; i < Args.size(); ++i) {
      auto tmp = pointeeComparisonWithOutput(Args[i], rhs.Args[i]);
      if(tmp.second == false) {
        output += "mismatch of argument " + Args[i]->Name + "\n" + tmp.first;
      }
    }
    return std::make_pair(output, false);
  }

  if(!Intervals.empty() &&
     !std::equal(Intervals.begin(), Intervals.end(), rhs.Intervals.begin(),
                 pointeeComparison<sir::Interval>)) {
    output += "[StencilFunction mismatch] Intervals do not match\n";
    for(unsigned i = 0; i < Intervals.size(); ++i) {
      auto tmp = pointeeComparisonWithOutput(Intervals[i], rhs.Intervals[i]);
      if(tmp.second == false) {
        output += tmp.first;
        return std::make_pair(output, false);
      }
    }
    return std::make_pair(output, false);
  }

  if(!Asts.empty()) {
    auto astcomparison = [](const std::shared_ptr<dawn::AST>& comparate1,
                            const std::shared_ptr<dawn::AST>& comparate2) {
      return *(comparate1->getRoot()) == *(comparate2->getRoot());
    };
    if(!std::equal(Asts.begin(), Asts.end(), rhs.Asts.begin(), astcomparison)) {
      output += "[StencilFunction mismatch] ASTs do not match\n";
      for(int i = 0; i < Asts.size(); ++i) {
        auto out = compareAst(Asts[i], rhs.Asts[i]);
        if(!out.second) {
          output += out.first;
          return std::make_pair(output, false);
        }
      }
    }
  }
  return std::make_pair(output, true);
}

std::pair<std::string, bool>
sir::StencilFunctionArg::comparison(const sir::StencilFunctionArg& rhs) const {
  std::string output;
  if(Name != rhs.Name) {
    output += dawn::format("[StencilFunctionArgument mismatch] Names do not match\n"
                           "Found Names %s and %s",
                           Name, rhs.Name);
    return std::make_pair(output, false);
  }
  if(Kind != rhs.Kind) {
    output += dawn::format("[StencilFunctionArgument mismatch] Argument Types do not match\n"
                           "Found Kinds %i and %i",
                           (int)Kind, (int)rhs.Kind);
    return std::make_pair(output, false);
  }
  return std::make_pair(output, true);
}

std::pair<std::string, bool> sir::Value::comparison(const sir::Value& rhs) const {

  std::string output;
  auto type = getType();
  if(type != rhs.getType()) {
    output += dawn::format("[Value mismatch] Values are not of the same type\n"
                           "Found Types %i and %i",
                           type, rhs.getType());
    return std::make_pair(output, false);
  }
  switch(type) {
  case sir::Value::TypeKind::Boolean:
    return isEqualImpl<bool>(*this, rhs, rhs.toString());
  case sir::Value::TypeKind::Integer:
    return isEqualImpl<int>(*this, rhs, rhs.toString());
  case sir::Value::TypeKind::Double:
    return isEqualImpl<double>(*this, rhs, rhs.toString());
  case sir::Value::TypeKind::String:
    return isEqualImpl<std::string>(*this, rhs, rhs.toString());
  default:
    dawn_unreachable("invalid type");
  }
}

std::pair<std::string, bool> sir::VerticalRegion::comparison(const sir::VerticalRegion& rhs) const {
  std::string output;
  if(LoopOrder != rhs.LoopOrder) {
    output += dawn::format("[VerticalRegion mismatch] Loop order does not match"
                           "loop orders are %i and %i",
                           LoopOrder, rhs.LoopOrder);
    return std::make_pair(output, false);
  }
  auto tmp = VerticalInterval->comparison(*(rhs.VerticalInterval));
  if(!tmp.second) {
    output += "[VerticalRegion mismatch] Intervals do not match\n";
    output += tmp.first;
    return std::make_pair(output, false);
  }
  auto astcomp = compareAst(Ast, rhs.Ast);
  if(!astcomp.second) {
    output += "[VerticalRegion mismatch] ASTs do not match\n";
    output += astcomp.first;
    return std::make_pair(output, false);
  }
  return std::make_pair("", true);
}

bool sir::VerticalRegion::operator==(const sir::VerticalRegion& rhs) const {
  return this->comparison(rhs).second;
}

namespace sir {

bool StencilFunction::isSpecialized() const { return !Intervals.empty(); }

std::shared_ptr<AST> StencilFunction::getASTOfInterval(const Interval& interval) const {
  for(int i = 0; i < Intervals.size(); ++i)
    if(*Intervals[i] == interval)
      return Asts[i];
  return nullptr;
}

std::string Interval::toString() const {
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

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

Stencil::Stencil() : StencilDescAst(std::make_shared<AST>()) {}

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

SIR::SIR() : GlobalVariableMap(std::make_shared<sir::GlobalVariableMap>()) {}

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

bool SIR::operator==(const SIR& rhs) const { return comparison(rhs).second; }

bool SIR::operator!=(const SIR& rhs) const { return !(*this == rhs); }

bool sir::Stencil::operator==(const sir::Stencil& rhs) const { return comparison(rhs).second; }

bool sir::StencilFunction::operator==(const sir::StencilFunction& rhs) const {
  return comparison(rhs).second;
}

bool sir::StencilFunctionArg::operator==(const sir::StencilFunctionArg& rhs) const {
  return comparison(rhs).second;
}

bool sir::Value::operator==(const sir::Value& rhs) const { return comparison(rhs).second; }

} // namespace dawn
