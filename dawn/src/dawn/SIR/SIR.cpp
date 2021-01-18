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

#include "dawn/AST/ASTStringifier.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Printing.h"

namespace dawn {

namespace {

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
CompareResult pointeeComparisonWithOutput(const std::shared_ptr<T>& comparate1,
                                          const std::shared_ptr<T>& comparate2) {
  return (*comparate1).comparison(*comparate2);
}

/// @brief Helperfunction to compare two global maps
///
/// the boolean is true if contents of the shared pointers match for every key (operator ==)
/// the string returns a potential mismatch notification
///
/// @return pair of boolean and string
static std::pair<std::string, bool> globalMapComparison(const ast::GlobalVariableMap& map1,
                                                        const ast::GlobalVariableMap& map2) {
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
      } else if(!(finder->second == a.second)) {
        output +=
            dawn::format("[GlobalVariableMap mismatch] Global Variables '%s' values are not equal\n"
                         "  Actual:\n"
                         "    %s\n"
                         "  Expected:\n"
                         "    %s",
                         a.first, a.second.toString(), finder->second.toString());
        return std::make_pair(output, false);
      }
    }
    return std::make_pair(output, true);
  }
}

} // anonymous namespace

CompareResult SIR::comparison(const SIR& rhs) const {
  std::string output;

  if(GridType != rhs.GridType)
    return CompareResult{"[SIR mismatch] grid type differs\n", false};

  // Stencils
  if((Stencils.size() != rhs.Stencils.size()))
    return CompareResult{"[SIR mismatch] number of Stencils do not match\n", false};

  if(!std::equal(Stencils.begin(), Stencils.end(), rhs.Stencils.begin(),
                 pointeeComparison<sir::Stencil>)) {
    output += "[SIR mismatch] Stencils do not match\n";
    for(int i = 0; i < Stencils.size(); ++i) {
      auto comp = pointeeComparisonWithOutput(Stencils[i], rhs.Stencils[i]);
      if(bool(comp) == false) {
        output += comp.why();
      }
    }

    return CompareResult{output, false};
  }

  // Stencil Functions
  if(StencilFunctions.size() != rhs.StencilFunctions.size())
    return CompareResult{"[SIR mismatch] number of Stencil Functions does not match\n", false};

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

    return CompareResult{output, false};
  }

  // Global variable map
  if(GlobalVariableMap.get()->size() != rhs.GlobalVariableMap.get()->size())
    return CompareResult{"[SIR mismatch] number of Global Variables does not match\n", false};

  if(!globalMapComparison(*(GlobalVariableMap.get()), *(rhs.GlobalVariableMap.get())).second) {
    auto comp = globalMapComparison(*(GlobalVariableMap.get()), *(rhs.GlobalVariableMap.get()));
    if(!comp.second)
      return CompareResult{comp.first, false};
  }

  return CompareResult{"", true};
}

CompareResult sir::Stencil::comparison(const sir::Stencil& rhs) const {
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

CompareResult sir::StencilFunction::comparison(const sir::StencilFunction& rhs) const {

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
                                       pointeeComparison<ast::Interval>)) {
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

  auto intervalToString = [](const ast::Interval& interval) {
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

CompareResult sir::StencilFunctionArg::comparison(const sir::StencilFunctionArg& rhs) const {
  auto kindToString = [](ArgumentKind kind) -> const char* {
    switch(kind) {
    case dawn::sir::StencilFunctionArg::ArgumentKind::Field:
      return "Field";
    case dawn::sir::StencilFunctionArg::ArgumentKind::Direction:
      return "Direction";
    case dawn::sir::StencilFunctionArg::ArgumentKind::Offset:
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

namespace sir {

std::shared_ptr<ast::AST> StencilFunction::getASTOfInterval(const ast::Interval& interval) const {
  for(int i = 0; i < Intervals.size(); ++i)
    if(*Intervals[i] == interval)
      return Asts[i];
  return nullptr;
}

bool StencilFunction::isSpecialized() const { return !Intervals.empty(); }

Stencil::Stencil() : StencilDescAst(sir::makeAST()) {}

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

UnstructuredFieldDimension::UnstructuredFieldDimension(ast::NeighborChain neighborChain,
                                                       bool includeCenter)
    : iterSpace_(std::move(neighborChain), includeCenter) {}

const ast::NeighborChain& UnstructuredFieldDimension::getNeighborChain() const {
  DAWN_ASSERT(isSparse());
  return iterSpace_.Chain;
}

std::string UnstructuredFieldDimension::toString() const {
  auto getLocationTypeString = [](const ast::LocationType type) {
    switch(type) {
    case ast::LocationType::Cells:
      return std::string("cell");
      break;
    case ast::LocationType::Vertices:
      return std::string("vertex");
      break;
    case ast::LocationType::Edges:
      return std::string("edge");
      break;
    default:
      dawn_unreachable("unexpected type");
    }
  };

  std::string output = "", separator = "";
  for(const auto elem : iterSpace_.Chain) {
    if(iterSpace_.IncludeCenter && separator == "") {
      output += separator + "[" + getLocationTypeString(elem) + "]";
    } else {
      output += separator + getLocationTypeString(elem);
    }
    separator = "->";
  }
  return output;
}

ast::GridType HorizontalFieldDimension::getType() const {
  if(sir::dimension_isa<sir::CartesianFieldDimension>(*this)) {
    return ast::GridType::Cartesian;
  } else {
    return ast::GridType::Unstructured;
  }
}

std::string FieldDimensions::toString() const {
  if(sir::dimension_isa<sir::CartesianFieldDimension>(getHorizontalFieldDimension())) {
    const auto& cartesianDimensions =
        sir::dimension_cast<sir::CartesianFieldDimension const&>(getHorizontalFieldDimension());
    return format("[%i,%i,%i]", cartesianDimensions.I(), cartesianDimensions.J(), K());

  } else if(sir::dimension_isa<sir::UnstructuredFieldDimension>(getHorizontalFieldDimension())) {
    const auto& unstructuredDimension =
        sir::dimension_cast<sir::UnstructuredFieldDimension const&>(getHorizontalFieldDimension());
    return format("[%s,%i]", unstructuredDimension.toString(), K());

  } else {
    dawn_unreachable("Invalid horizontal field dimension");
  }
}

int FieldDimensions::numSpatialDimensions() const {
  if(!horizontalFieldDimension_) {
    return 1;
  }
  if(sir::dimension_isa<sir::CartesianFieldDimension>(getHorizontalFieldDimension())) {
    const auto& cartesianDimensions =
        sir::dimension_cast<sir::CartesianFieldDimension const&>(getHorizontalFieldDimension());
    return int(cartesianDimensions.I()) + int(cartesianDimensions.J()) + int(K());
  } else if(sir::dimension_isa<sir::UnstructuredFieldDimension>(getHorizontalFieldDimension())) {
    return 2 + int(K());
  } else {
    dawn_unreachable("Invalid horizontal field dimension");
  }
}

int FieldDimensions::rank() const {
  const int spatialDims = numSpatialDimensions();
  if(isVertical()) {
    return 1;
  }
  int rank;
  if(sir::dimension_isa<sir::UnstructuredFieldDimension>(getHorizontalFieldDimension())) {
    rank = spatialDims > 1 ? spatialDims - 1 // The horizontal counts as 1 dimension (dense)
                           : spatialDims;
    // Need to account for sparse dimension, if present
    if(sir::dimension_cast<sir::UnstructuredFieldDimension const&>(getHorizontalFieldDimension())
           .isSparse()) {
      ++rank;
    }
  } else { // Cartesian
    rank = spatialDims;
  }
  return rank;
}

bool Stencil::operator==(const sir::Stencil& rhs) const { return bool(comparison(rhs)); }

bool StencilFunction::operator==(const sir::StencilFunction& rhs) const {
  return bool(comparison(rhs));
}

bool StencilFunctionArg::operator==(const sir::StencilFunctionArg& rhs) const {
  return bool(comparison(rhs));
}

} // namespace sir

std::ostream& operator<<(std::ostream& os, const SIR& Sir) {
  const char* indent1 = MakeIndent<1>::value;
  const char* indent2 = MakeIndent<2>::value;

  os << "SIR : " << Sir.Filename << "\n{\n";

  os << "Grid type : " << Sir.GridType << "\n";

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
      os << ast::ASTStringifier::toString(*stencilFunction->Asts[0], 2 * DAWN_PRINT_INDENT);
    } else {
      for(int i = 0; i < stencilFunction->Intervals.size(); ++i) {
        os << "\n" << indent2 << "Do " << *stencilFunction->Intervals[i].get() << "\n";
        os << ast::ASTStringifier::toString(*stencilFunction->Asts[i], 2 * DAWN_PRINT_INDENT);
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
       << ast::ASTStringifier::toString(*stencil->StencilDescAst, 2 * DAWN_PRINT_INDENT);
    os << indent1 << "}\n";
  }

  os << "\n}";
  return os;
}

SIR::SIR(const ast::GridType gridType)
    : GlobalVariableMap(std::make_shared<ast::GlobalVariableMap>()), GridType(gridType) {}

void SIR::dump(std::ostream& os) { os << *this << std::endl; }

bool SIR::operator==(const SIR& rhs) const { return comparison(rhs); }

bool SIR::operator!=(const SIR& rhs) const { return !(*this == rhs); }

} // namespace dawn
