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
class DiffWriter : public ASTVisitorForwarding {
public:
  DiffWriter()
      : expressions(std::make_shared<std::vector<std::shared_ptr<Expr>>>()),
        statements(std::make_shared<std::vector<std::shared_ptr<Stmt>>>()) {}

  virtual void visit(const std::shared_ptr<UnaryOperator>& expr) override {
    expressions->push_back(expr);
    ASTVisitorForwarding::visit(expr);
  }
  virtual void visit(const std::shared_ptr<BinaryOperator>& expr) override {
    expressions->push_back(expr);
    ASTVisitorForwarding::visit(expr);
  }
  virtual void visit(const std::shared_ptr<TernaryOperator>& expr) override {
    expressions->push_back(expr);
    ASTVisitorForwarding::visit(expr);
  }
  virtual void visit(const std::shared_ptr<FunCallExpr>& expr) override {
    expressions->push_back(expr);
    ASTVisitorForwarding::visit(expr);
  }
  virtual void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override {
    expressions->push_back(expr);
    ASTVisitorForwarding::visit(expr);
  }
  virtual void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override {
    expressions->push_back(expr);
    ASTVisitorForwarding::visit(expr);
  }

  virtual void visit(const std::shared_ptr<AssignmentExpr>& expr) override {
    expressions->push_back(expr);
    ASTVisitorForwarding::visit(expr);
  }
  virtual void visit(const std::shared_ptr<VarAccessExpr>& expr) override {
    expressions->push_back(expr);
    ASTVisitorForwarding::visit(expr);
  }

  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {
    expressions->push_back(expr);
    ASTVisitorForwarding::visit(expr);
  }
  virtual void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) override {
    statements->push_back(stmt);
    stmt->getVerticalRegion()->Ast->getRoot()->accept(*this);
  }

  virtual void visit(const std::shared_ptr<ReturnStmt>& stmt) override {
    statements->push_back(stmt);
    ASTVisitorForwarding::visit(stmt);
  }

  virtual void visit(const std::shared_ptr<ExprStmt>& stmt) override {
    statements->push_back(stmt);
    ASTVisitorForwarding::visit(stmt);
  }

  virtual void visit(const std::shared_ptr<BlockStmt>& stmt) override {
    statements->push_back(stmt);
    ASTVisitorForwarding::visit(stmt);
  }
  virtual void visit(const std::shared_ptr<VarDeclStmt>& stmt) override {
    statements->push_back(stmt);
    ASTVisitorForwarding::visit(stmt);
  }

  std::shared_ptr<std::vector<std::shared_ptr<Expr>>> getExpr() { return expressions; }
  std::shared_ptr<std::vector<std::shared_ptr<Stmt>>> getStmt() { return statements; }

private:
  std::shared_ptr<std::vector<std::shared_ptr<Expr>>> expressions;
  std::shared_ptr<std::vector<std::shared_ptr<Stmt>>> statements;
};

std::pair<std::string, bool> compareAst(std::shared_ptr<AST> lhs, std::shared_ptr<AST> rhs) {
  std::string output;
  if(lhs->getRoot()->getStatements().size() != rhs->getRoot()->getStatements().size()) {
    output += "[AST missmatch] ASTs do not have the same number of statements";
    return std::make_pair(output, false);
  }
  for(unsigned i = 0; i < lhs->getRoot()->getStatements().size(); ++i) {
    if(!lhs->getRoot()->getStatements()[i]->equals(rhs->getRoot()->getStatements()[i].get())) {
      auto a = lhs->getRoot()->getStatements()[i];
      auto b = rhs->getRoot()->getStatements()[i];
      auto kind = a->getKind();
      auto rhskind = b->getKind();
      if(kind != rhskind) {
        output += "[AST missmatch] Statements are not of the same kind";
        return std::make_pair(output, false);
      }
      DiffWriter dw;
      a->accept(dw);
      auto lhsExpr = dw.getExpr();
      auto lhsStmt = dw.getStmt();
      DiffWriter dw2;
      b->accept(dw2);
      auto rhsExpr = dw2.getExpr();
      auto rhsStmt = dw2.getStmt();
      if(lhsExpr->size() != rhsExpr->size() || lhsStmt->size() != rhsStmt->size()) {
        output += "[AST missmatch] Number of Statements / Expressions does not match";
        return std::make_pair(output, false);
      }
      for(unsigned j = 0; j < lhsExpr->size(); ++j) {
        if(lhsExpr->at(j) != rhsExpr->at(j)) {
          output += dawn::format("[AST missmatch] Uneven Statements\n"
                                 "[Expression missmatch] Expecting \n %s \n and got \n %s \n",
                                 ASTStringifer::toString(lhsExpr->at(j)),
                                 ASTStringifer::toString(rhsExpr->at(j)));
          return std::make_pair(output, false);
        }
      }
      for(unsigned j = 0; j < lhsStmt->size(); ++j) {
        if(lhsStmt->at(j) != rhsStmt->at(j)) {
          output += dawn::format("[AST missmatch] Uneven Statements\n"
                                 "[Statement missmatch] Expecting \n %s \n and got \n %s \n",
                                 ASTStringifer::toString(lhsStmt->at(j)),
                                 ASTStringifer::toString(rhsStmt->at(j)));
          return std::make_pair(output, false);
        }
      }
    }
  }
  return std::make_pair(output, true);
}

/// @brief Compares the content of two shared pointers
/// @param[in] shared pointer of type T
/// @param[in] shared pointer of same type T
/// @return true if contents of the shared pointers match (operator ==)
template <typename T>
bool pointeeComparison(const std::shared_ptr<T>& comparate1, const std::shared_ptr<T>& comparate2) {
  return *comparate1 == *comparate2;
}

/// @brief Compares the content of two shared pointers
/// @param[in] shared pointer of type T
/// @param[in] shared pointer of same type T
/// @return pair of boolean and string
/// the boolean is true if contents of the shared pointers match (operator ==)
/// the string returns a potential missmatch notification
/// @pre Type T requies a comparison function that returns the pair of bool and string
template <typename T>
std::pair<std::string, bool> pointeeComparisonWithOutput(const std::shared_ptr<T>& comparate1,
                                                         const std::shared_ptr<T>& comparate2) {
  return (*comparate1).comparison(*comparate2);
}

static bool pointerMapComparison(const sir::GlobalVariableMap& map1,
                                 const sir::GlobalVariableMap& map2) {
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

std::pair<std::string, bool> pointerMapComparisonWithOutput(const sir::GlobalVariableMap& map1,
                                                            const sir::GlobalVariableMap& map2) {
  std::string output;
  if(map1.size() != map2.size()) {
    output += "[GlobalVariableMap missmatch] unequal number of variables defined\n";
    return std::make_pair(output, false);
  } else {
    for(auto& a : map1) {
      auto finder = map2.find(a.first);
      if(finder == map2.end()) {
        output += "[GlobalVariableMap missmatch] element " + a.first + " only found in one map\n";
        return std::make_pair(output, false);
      } else if(finder->second == nullptr || a.second == nullptr) {
        return std::make_pair("", finder->second == a.second);
      } else if(!(*(finder->second.get()) == *(a.second.get()))) {
        output += "[GlobalVariableMap missmatch] element " + a.first + " has different values\n";
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
    return std::make_pair("[SIR missmatch] number of stencils do not match\n", false);
  if(StencilFunctions.size() != rhs.StencilFunctions.size())
    return std::make_pair("[SIR missmatch] number of stencil functions does not match\n", false);
  if(GlobalVariableMap.get()->size() != rhs.GlobalVariableMap.get()->size())
    return std::make_pair("[SIR missmatch] number of global variables does not match\n", false);

  if(!Stencils.empty() &&
     !std::equal(Stencils.begin(), Stencils.end(), rhs.Stencils.begin(),
                 pointeeComparison<sir::Stencil>)) {
    output += "[SIR missmatch] stencils do not match\n";
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
    output += "[SIR missmatch] stencil functions do not match\n";
    for(unsigned i = 0; i < StencilFunctions.size(); ++i) {
      auto tmp = pointeeComparisonWithOutput(StencilFunctions[i], rhs.StencilFunctions[i]);
      if(tmp.second == false) {
        output += "Missmatch of function " + StencilFunctions[i]->Name + "\n" + tmp.first;
      }
    }
    return std::make_pair(output, false);
  }
  if(!GlobalVariableMap.get()->empty() &&
     !pointerMapComparison(*(GlobalVariableMap.get()), *(rhs.GlobalVariableMap.get()))) {
    auto a = pointerMapComparisonWithOutput(*GlobalVariableMap.get(), *rhs.GlobalVariableMap.get());
    return std::make_pair(a.first, false);
  }

  return std::make_pair("", true);
}

std::pair<std::string, bool> sir::Stencil::comparison(const sir::Stencil& rhs) const {
  std::string output;
  if(Fields.size() != rhs.Fields.size()) {
    output += "[Stencil missmatch] number of Fields does not match\n";
    return std::make_pair(output, false);
  }
  if(Name != rhs.Name) {
    output += "[Stencil missmatch] Stencil names do not match\n";
    return std::make_pair(output, false);
  }
  if(!(Attributes == rhs.Attributes)) {
    output += "[Stencil missmatch] Stencil attibutes do not match\n";
    output += dawn::format("of Stencil %s and %s", Name, rhs.Name);
    return std::make_pair(output, false);
  }
  if(!StencilDescAst->getRoot().get()->equals(rhs.StencilDescAst->getRoot().get())) {
    output += "[Stencil missmatch] Stencil ASTs Do not match\n";
    auto a = compareAst(StencilDescAst, rhs.StencilDescAst);
    if(a.second)
      dawn_unreachable("Stencils should not match");
    else {
      output += a.first;
      return std::make_pair(output, false);
    }
  }
  if(!Fields.empty() &&
     !std::equal(Fields.begin(), Fields.end(), rhs.Fields.begin(), pointeeComparison<sir::Field>)) {
    output += "[Stencil missmatch] Fields do not match\n";
    for(unsigned i = 0; i < Fields.size(); ++i) {
      auto tmp = pointeeComparisonWithOutput(Fields[i], rhs.Fields[i]);
      if(tmp.second == false) {
        output += "Field " + Fields[i].get()->Name + " missmatch: " + tmp.first;
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
    output += "[StencilFunction missmatch] Names of Stencil Functions do not match\n";
    return std::make_pair(output, false);
  }
  if(!(Attributes == rhs.Attributes)) {
    output += "[StencilFunction missmatch] Attributes of Stencil Functions do not match\n";
    return std::make_pair(output, false);
  }
  if(Args.size() != rhs.Args.size()) {
    output += "[StencilFunction missmatch] Number of Arguments do not match\n";
    return std::make_pair(output, false);
  }
  if(Intervals.size() != rhs.Intervals.size()) {
    output += "[StencilFunction missmatch] Number of Intervals do not match\n";
    return std::make_pair(output, false);
  }
  if(Asts.size() != rhs.Asts.size()) {
    output += "[StencilFunction missmatch] Size of ASTs does not match\n";
    return std::make_pair(output, false);
  }
  if(!Args.empty() &&
     !std::equal(Args.begin(), Args.end(), rhs.Args.begin(),
                 pointeeComparison<sir::StencilFunctionArg>)) {
    output += "[StencilFunction missmatch] stencil functions arguments do not match\n";
    for(unsigned i = 0; i < Args.size(); ++i) {
      auto tmp = pointeeComparisonWithOutput(Args[i], rhs.Args[i]);
      if(tmp.second == false) {
        output += "Missmatch of argument " + Args[i]->Name + " " + tmp.first;
      }
    }
    return std::make_pair(output, false);
  }

  if(!Intervals.empty() &&
     !std::equal(Intervals.begin(), Intervals.end(), rhs.Intervals.begin(),
                 pointeeComparison<sir::Interval>)) {
    output += "[StencilFunction missmatch] stencil functions intervals do not match\n";
    for(unsigned i = 0; i < Intervals.size(); ++i) {
      auto tmp = pointeeComparison(Intervals[i], rhs.Intervals[i]);
      if(tmp == false) {
        output += "Missmatch of intervals " + Intervals[i]->toString() + " and " +
                  rhs.Intervals[i]->toString() + "\n";
      }
    }
    return std::make_pair(output, false);
  }

  if(!Asts.empty()) {
    auto astcomparison = [](std::shared_ptr<dawn::AST> comparate1,
                            std::shared_ptr<dawn::AST> comparate2) {
      return *(comparate1->getRoot()) == *(comparate2->getRoot());
    };
    if(!std::equal(Asts.begin(), Asts.end(), rhs.Asts.begin(), astcomparison)) {
      output += "[StencilFunction missmatch] ASTs do not match\n";
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
    output += "[StencilFunctionArgument missmatch] Names do not match\n";
    return std::make_pair(output, false);
  }
  if(Kind != rhs.Kind) {
    output += "[StencilFunctionArgument missmatch] Argument Types do not match\n";
    return std::make_pair(output, false);
  }
  return std::make_pair(output, true);
}

std::pair<std::string, bool> sir::Value::comparison(const sir::Value& rhs) const {

  std::string output;
  auto type = getType();
  if(type != rhs.getType()) {
    output += "[Value missmatch] Values are not of the same type\n";
    return std::make_pair(output, false);
  }
  switch(type) {
  case sir::Value::TypeKind::Boolean:
    if(getValue<bool>() != rhs.getValue<bool>()) {
      output += "[Value missmatch] Boolean Values are not equal\n";
      return std::make_pair(output, false);
    } else
      return std::make_pair(output, true);
  case sir::Value::TypeKind::Integer:
    if(getValue<int>() != rhs.getValue<int>()) {
      output += "[Value missmatch] Integer Values are not equal\n";
      return std::make_pair(output, false);
    } else
      return std::make_pair(output, true);
  case sir::Value::TypeKind::Double:
    if(getValue<double>() != rhs.getValue<double>()) {
      output += "[Value missmatch] Double Values are not equal\n";
      return std::make_pair(output, false);
    } else
      return std::make_pair(output, true);
  case sir::Value::TypeKind::String:
    if(getValue<std::string>() != rhs.getValue<std::string>()) {
      output += "[Value missmatch] Boolean Values are not equal\n";
      return std::make_pair(output, false);
    } else
      return std::make_pair(output, true);
  default:
    dawn_unreachable("invalid type");
  }
}

std::pair<std::string, bool> sir::VerticalRegion::comparison(const sir::VerticalRegion& rhs) const {
  std::string output;
  if(LoopOrder != rhs.LoopOrder) {
    output += "[VerticalRegion missmatch] Loop order does not match";
    return std::make_pair(output, false);
  }
  auto tmp = VerticalInterval->comparison(*(rhs.VerticalInterval));
  if(!tmp.second) {
    output += "[VerticalRegion missmatch] Intervals do not match\n";
    output += tmp.first;
    return std::make_pair(output, false);
  }
  auto astcomp = compareAst(Ast, rhs.Ast);
  if(!astcomp.second){
      output += "[VerticalRegion missmatch] ASTs do not match\n";
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
