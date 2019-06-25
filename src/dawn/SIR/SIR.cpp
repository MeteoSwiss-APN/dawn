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

  virtual void visit(const std::shared_ptr<IfStmt>& stmt) override {
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
sir::CompareResult isEqualImpl(const sir::Value& a, const sir::Value& b, const std::string& name) {
  if(a.getValue<T>() != b.getValue<T>())
    return sir::CompareResult{dawn::format("[Value mismatch] %s values are not equal\n"
                                           "  Actual:\n"
                                           "    %s\n"
                                           "  Expected:\n"
                                           "    %s",
                                           name, a.toString(), b.toString()),
                              false};

  return sir::CompareResult{"", true};
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
sir::CompareResult pointeeComparisonWithOutput(const std::shared_ptr<T>& comparate1,
                                               const std::shared_ptr<T>& comparate2) {
  return (*comparate1).comparison(*comparate2);
}

/// @brief Helperfunction to compare two maps of key and shared pointer
///
/// the boolean is true if contents of the shared pointers match for every key (operator ==)
/// the string returns a potential mismatch notification
///
/// @return pair of boolean and string
static std::pair<std::string, bool> pointerMapComparison(const sir::GlobalVariableMap& map1,
                                                         const sir::GlobalVariableMap& map2) {
  std::string output;
  if(map1.size() != map2.size()) {
    output += dawn::format("[GlobalVariableMap mismatch] Number of Global Variables do not match\n"
                           "  Actual:\n"
                           "    %s\n"
                           "  Expected:\n"
                           "    %s",
                           map1.size(), map2.size());
    return std::make_pair(output, false);

  } else {
    for(auto& a : map1) {
      auto finder = map2.find(a.first);
      if(finder == map2.end()) {
        output +=
            dawn::format("[GlobalVariableMap mismatch] Global Variable '%s' not found\n", a.first);
        return std::make_pair(output, false);
      } else if(!(*(finder->second.get()) == *(a.second.get()))) {
        output +=
            dawn::format("[GlobalVariableMap mismatch] Global Variables '%s' values are not equal\n"
                         "  Actual:\n"
                         "    %s\n"
                         "  Expected:\n"
                         "    %s",
                         a.first, a.second->toString(), finder->second->toString());
        return std::make_pair(output, false);
      }
    }
    return std::make_pair(output, true);
  }
}

} // anonymous namespace

sir::CompareResult SIR::comparison(const SIR& rhs) const {
  std::string output;

  // Stencils
  if((Stencils.size() != rhs.Stencils.size()))
    return sir::CompareResult{"[SIR mismatch] number of Stencils do not match\n", false};

  if(!std::equal(Stencils.begin(), Stencils.end(), rhs.Stencils.begin(),
                 pointeeComparison<sir::Stencil>)) {
    output += "[SIR mismatch] Stencils do not match\n";
    for(int i = 0; i < Stencils.size(); ++i) {
      auto comp = pointeeComparisonWithOutput(Stencils[i], rhs.Stencils[i]);
      if(bool(comp) == false) {
        output += comp.why();
      }
    }

    return sir::CompareResult{output, false};
  }

  // Stencil Functions
  if(StencilFunctions.size() != rhs.StencilFunctions.size())
    return sir::CompareResult{"[SIR mismatch] number of Stencil Functions does not match\n", false};

  if(!std::equal(StencilFunctions.begin(), StencilFunctions.end(), rhs.StencilFunctions.begin(),
                 pointeeComparison<sir::StencilFunction>)) {
    output += "[SIR mismatch] Stencil Functions do not match\n";
    for(int i = 0; i < StencilFunctions.size(); ++i) {
      auto comp = pointeeComparisonWithOutput(StencilFunctions[i], rhs.StencilFunctions[i]);
      if(!bool(comp))
        output +=
            dawn::format("[StencilFunction mismatch] Stencil Function %s does not match\n%s\n",
                         StencilFunctions[i]->Name, comp.why());
    }

    return sir::CompareResult{output, false};
  }

  // Global variable map
  if(GlobalVariableMap.get()->size() != rhs.GlobalVariableMap.get()->size())
    return sir::CompareResult{"[SIR mismatch] number of Global Variables does not match\n", false};

  if(!pointerMapComparison(*(GlobalVariableMap.get()), *(rhs.GlobalVariableMap.get())).second) {
    auto comp = pointerMapComparison(*(GlobalVariableMap.get()), *(rhs.GlobalVariableMap.get()));
    if(!comp.second)
      return sir::CompareResult{comp.first, false};
  }

  return sir::CompareResult{"", true};
}

sir::CompareResult sir::Stencil::comparison(const sir::Stencil& rhs) const {
  // Fields
  if(Fields.size() != rhs.Fields.size())
    return CompareResult{dawn::format("[Stencil mismatch] number of Fields does not match\n"
                                      "  Actual:\n"
                                      "    %s\n"
                                      "  Expected:\n"
                                      "    %s",
                                      Fields.size(), rhs.Fields.size()),
                         false};

  // Name
  if(Name != rhs.Name)
    return CompareResult{dawn::format("[Stencil mismatch] Stencil names do not match\n"
                                      "  Actual:\n"
                                      "    %s\n"
                                      "  Expected:\n"
                                      "    %s",
                                      Name, rhs.Name),
                         false};

  if(!Fields.empty() && !std::equal(Fields.begin(), Fields.end(), rhs.Fields.begin(),
                                    pointeeComparisonWithOutput<Field>)) {
    std::string output = "[Stencil mismatch] Fields do not match\n";
    for(int i = 0; i < Fields.size(); ++i) {
      auto comp = pointeeComparisonWithOutput(Fields[i], rhs.Fields[i]);
      if(!comp) {
        output += dawn::format("[Stencil mismatch] Field %s does not match\n%s\n",
                               Fields[i].get()->Name, comp.why());
      }
    }

    return CompareResult{output, false};
  }

  // AST
  auto astComp = compareAst(StencilDescAst, rhs.StencilDescAst);
  if(!astComp.second)
    return CompareResult{dawn::format("[Stencil mismatch] ASTs do not match\n%s\n", astComp.first),
                         false};

  return CompareResult{"", true};
}

sir::CompareResult sir::StencilFunction::comparison(const sir::StencilFunction& rhs) const {

  // Name
  if(Name != rhs.Name) {
    return CompareResult{
        dawn::format("[StencilFunction mismatch] Names of Stencil Functions do not match\n"
                     "  Actual:\n"
                     "    %s\n"
                     "  Expected:\n"
                     "    %s",
                     Name, rhs.Name),
        false};
  }

  // Arguments
  if(Args.size() != rhs.Args.size()) {
    return CompareResult{
        dawn::format("[StencilFunction mismatch] Number of Arguments do not match\n"
                     "  Actual:\n"
                     "    %i\n"
                     "  Expected:\n"
                     "    %i",
                     Args.size(), rhs.Args.size()),
        false};
  }

  if(!std::equal(Args.begin(), Args.end(), rhs.Args.begin(),
                 pointeeComparison<sir::StencilFunctionArg>)) {
    std::string output = "[StencilFunction mismatch] Stencil Functions Arguments do not match\n";
    for(int i = 0; i < Args.size(); ++i) {
      auto comp = pointeeComparisonWithOutput(Args[i], rhs.Args[i]);
      if(!bool(comp)) {
        output += dawn::format("[StencilFunction mismatch] Argument '%s' does not match\n%s\n",
                               Args[i]->Name, comp.why());
      }
    }
    return CompareResult{output, false};
  }

  // Intervals
  if(Intervals.size() != rhs.Intervals.size()) {
    return CompareResult{
        dawn::format("[StencilFunction mismatch] Number of Intervals do not match\n"
                     "  Actual:\n"
                     "    %i\n"
                     "  Expected:\n"
                     "    %i",
                     Intervals.size(), rhs.Intervals.size()),
        false};
  }

  // Intervals
  if(!Intervals.empty() && !std::equal(Intervals.begin(), Intervals.end(), rhs.Intervals.begin(),
                                       pointeeComparison<sir::Interval>)) {
    std::string output = "[StencilFunction mismatch] Intervals do not match\n";
    for(int i = 0; i < Intervals.size(); ++i) {
      auto comp = pointeeComparisonWithOutput(Intervals[i], rhs.Intervals[i]);
      if(bool(comp) == false) {
        output += dawn::format("[StencilFunction mismatch] Interval '%s' does not match '%s'\n%s\n",
                               *Intervals[i], *rhs.Intervals[i], comp.why());
      }
    }
    return CompareResult{output, false};
  }

  // ASTs
  if(Asts.size() != rhs.Asts.size()) {
    return CompareResult{dawn::format("[StencilFunction mismatch] Number of ASTs does not match\n"
                                      "  Actual:\n"
                                      "    %i\n"
                                      "  Expected:\n"
                                      "    %i",
                                      Asts.size(), rhs.Asts.size()),
                         false};
  }

  auto intervalToString = [](const sir::Interval& interval) {
    std::stringstream ss;
    ss << interval;
    return ss.str();
  };

  for(int i = 0; i < Asts.size(); ++i) {
    auto astComp = compareAst(Asts[i], rhs.Asts[i]);
    if(!astComp.second)
      return CompareResult{
          dawn::format("[StencilFunction mismatch] AST '%s' does not match\n%s\n",
                       i < Intervals.size() ? intervalToString(*Intervals[i]) : std::to_string(i),
                       astComp.first),
          false};
  }

  return CompareResult{"", true};
}

sir::CompareResult sir::StencilFunctionArg::comparison(const sir::StencilFunctionArg& rhs) const {
  auto kindToString = [](ArgumentKind kind) -> const char* {
    switch(kind) {
    case dawn::sir::StencilFunctionArg::AK_Field:
      return "Field";
    case dawn::sir::StencilFunctionArg::AK_Direction:
      return "Direction";
    case dawn::sir::StencilFunctionArg::AK_Offset:
      return "Offset";
    }
    dawn_unreachable("invalid argument type");
  };

  if(Name != rhs.Name) {
    return CompareResult{dawn::format("[StencilFunctionArgument mismatch] Names do not match\n"
                                      "  Actual:\n"
                                      "    %s\n"
                                      "  Expected:\n"
                                      "    %s",
                                      Name, rhs.Name),
                         false};
  }

  if(Kind != rhs.Kind)
    return CompareResult{
        dawn::format("[StencilFunctionArgument mismatch] Argument Types do not match\n"
                     "  Actual:\n"
                     "    %s\n"
                     "  Expected:\n"
                     "    %s",
                     kindToString(Kind), kindToString(rhs.Kind)),
        false};

  return CompareResult{"", true};
}

sir::CompareResult sir::Value::comparison(const sir::Value& rhs) const {
  auto type = getType();
  if(type != rhs.getType())
    return CompareResult{dawn::format("[Value mismatch] Values are not of the same type\n"
                                      "  Actual:\n"
                                      "    %s\n"
                                      "  Expected:\n"
                                      "    %s",
                                      type, rhs.getType()),
                         false};

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

sir::CompareResult sir::VerticalRegion::comparison(const sir::VerticalRegion& rhs) const {
  std::string output;
  if(LoopOrder != rhs.LoopOrder) {
    output += dawn::format("[VerticalRegion mismatch] Loop order does not match\n"
                           "  Actual:\n"
                           "    %s\n"
                           "  Expected:\n"
                           "    %s",
                           LoopOrder, rhs.LoopOrder);
    return CompareResult{output, false};
  }

  auto intervalComp = VerticalInterval->comparison(*(rhs.VerticalInterval));
  if(!bool(intervalComp)) {
    output += "[VerticalRegion mismatch] Intervals do not match\n";
    output += intervalComp.why();
    return CompareResult{output, false};
  }

  auto astComp = compareAst(Ast, rhs.Ast);
  if(!astComp.second) {
    output += "[VerticalRegion mismatch] ASTs do not match\n";
    output += astComp.first;
    return CompareResult{output, false};
  }
  return CompareResult{output, true};
}

bool sir::VerticalRegion::operator==(const sir::VerticalRegion& rhs) const {
  return bool(this->comparison(rhs));
}

namespace sir {

bool StencilFunction::isSpecialized() const { return !Intervals.empty(); }

std::shared_ptr<AST> StencilFunction::getASTOfInterval(const Interval& interval) const {
  for(int i = 0; i < Intervals.size(); ++i)
    if(*Intervals[i] == interval)
      return Asts[i];
  return nullptr;
}

CompareResult Interval::comparison(const Interval& rhs) const {
  auto formatErrorMsg = [](const char* name, int l, int r) -> std::string {
    return dawn::format("[Inverval mismatch] %s do not match\n"
                        "  Actual:\n"
                        "    %i\n"
                        "  Expected:\n"
                        "    %i",
                        name, l, r);
  };

  if(LowerLevel != rhs.LowerLevel)
    return CompareResult{formatErrorMsg("LowerLevels", LowerLevel, rhs.LowerLevel), false};

  if(UpperLevel != rhs.UpperLevel)
    return CompareResult{formatErrorMsg("UpperLevels", UpperLevel, rhs.UpperLevel), false};

  if(LowerOffset != rhs.LowerOffset)
    return CompareResult{formatErrorMsg("LowerOffsets", LowerOffset, rhs.LowerOffset), false};

  if(UpperOffset != rhs.UpperOffset)
    return CompareResult{formatErrorMsg("UpperOffsets", UpperOffset, rhs.UpperOffset), false};

  return CompareResult{"", true};
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

CompareResult Field::comparison(const Field& rhs) const {
  if(rhs.IsTemporary != IsTemporary) {
    return {dawn::format("[Field Mismatch] Temporary Flags do not match"
                         "Actual:\n"
                         "%s\n"
                         "Expected:\n"
                         "%s",
                         (IsTemporary ? "true" : "false"), (rhs.IsTemporary ? "true" : "false")),
            false};
  }

  return StencilFunctionArg::comparison(rhs);
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
    return BuiltinTypeID::Invalid;
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

bool sir::StencilCall::operator==(const sir::StencilCall& rhs) const {
  return bool(this->comparison(rhs));
}

sir::CompareResult sir::StencilCall::comparison(const sir::StencilCall& rhs) const {
  std::string output;
  if(Callee != rhs.Callee) {
    output += dawn::format("[StencilCall mismatch] Callees do not match\n"
                           "  Actual:\n"
                           "    %s\n"
                           "  Expected:\n"
                           "    %s",
                           Callee, rhs.Callee);
    return CompareResult{output, false};
  }
  for(int i = 0; i < Args.size(); ++i) {
    auto ArgComparison = Args[i]->comparison(*rhs.Args[i]);
    if(!bool(ArgComparison)) {
      output += "[StencilCall mismatch] Arguments do not match\n";
      output += ArgComparison.why();
      return CompareResult{output, false};
    }
  }
  return CompareResult{output, true};
}

bool SIR::operator==(const SIR& rhs) const { return comparison(rhs); }

bool SIR::operator!=(const SIR& rhs) const { return !(*this == rhs); }

bool sir::Stencil::operator==(const sir::Stencil& rhs) const { return bool(comparison(rhs)); }

bool sir::StencilFunction::operator==(const sir::StencilFunction& rhs) const {
  return bool(comparison(rhs));
}

bool sir::StencilFunctionArg::operator==(const sir::StencilFunctionArg& rhs) const {
  return bool(comparison(rhs));
}

bool sir::Value::operator==(const sir::Value& rhs) const { return bool(comparison(rhs)); }

} // namespace dawn
