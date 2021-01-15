#include "dawn/SIR/SIR.h"
#include "dawn/AST/ASTStringifier.h"
#include "dawn/Support/Format.h"
#include <sstream>

namespace dawn {

namespace sir {

std::shared_ptr<ast::AST> StencilFunction::getASTOfInterval(const Interval& interval) const {
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

} // namespace sir
} // namespace dawn
