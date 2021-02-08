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

#pragma once

#include "dawn/Support/Assert.h"
#include <algorithm>
#include <functional>
#include <optional>
#include <sstream>
#include <vector>
#define DAWN_FORTRAN_INDENT 2

namespace dawn {
namespace codegen {

namespace {
void indent(int level, std::stringstream& ss) {
  ss << std::string(DAWN_FORTRAN_INDENT * level, ' ');
}
struct endlineType {};
static endlineType endline;
} // namespace

class IndentedStringStream {
protected:
  std::reference_wrapper<std::stringstream> ss_;
  int indentLevel_ = 0;
  bool freshNewLine_ = true;

public:
  IndentedStringStream(std::stringstream& s, int il = 0) : ss_(s), indentLevel_(il) {}

  IndentedStringStream& operator<<(endlineType) {
    ss() << std::endl;
    freshNewLine_ = true;
    return *this;
  }

  template <class T>
  IndentedStringStream& operator<<(T&& data) {
    if(freshNewLine_)
      indent(indentLevel_, ss());
    freshNewLine_ = false;
    ss() << data;
    return *this;
  }

  void increaseIndent() { indentLevel_++; }
  void decreaseIndent() {
    if(indentLevel_ > 0)
      indentLevel_--;
  }
  void setIndentLevel(int il) {
    DAWN_ASSERT(il >= 0);
    indentLevel_ = il;
  }

  std::stringstream& ss() { return ss_.get(); }

  std::string str() { return ss().str(); }
};

class FortranAPI {
public:
  enum class InterfaceType { INTEGER, FLOAT, DOUBLE, CHAR, BOOLEAN, OBJ };
  FortranAPI(std::string name, std::optional<InterfaceType> returnType = std::nullopt)
      : name_(name) {
    if(returnType) {
      returnType_ = TypeToString(*returnType);
    }
  }
  void addArg(std::string name, InterfaceType type, int dimensions = 0) {
    args_.push_back(std::make_tuple(name, TypeToString(type), dimensions));
  }

protected:
  static std::string TypeToString(InterfaceType t) {
    switch(t) {
    case InterfaceType::INTEGER:
      return "integer(c_int)";
    case InterfaceType::DOUBLE:
      return "real(c_double)";
    case InterfaceType::FLOAT:
      return "real(c_float)";
    case InterfaceType::CHAR:
      return "character(kind=c_char)";
    case InterfaceType::BOOLEAN:
      return "logical(c_bool)";
    case InterfaceType::OBJ:
      return "type(c_ptr)";
    }
    return "";
  }

  void streamAPISignature(IndentedStringStream& ss, bool isCBinding) const {
    bool isFunction = returnType_ != "";
    ss << (isFunction ? returnType_ + " function" : std::string("subroutine")) << " &" << endline;
    ss << name_ << "( ";
    if(args_.size() > 0) {
      ss << "&";
    }
    {
      std::string sep;
      for(const auto& arg : args_) {
        ss << sep << endline << std::get<0>(arg);
        sep = ", &";
      }
    }
    if(args_.size() > 0) {
      ss << " &" << endline;
    }
    ss << std::string(")") << (isCBinding ? " bind(c)" : "") << endline;
  }

  virtual void streamAPISignature(IndentedStringStream& ss) const = 0;

  virtual void streamArgsDecls(IndentedStringStream& ss) const = 0;

  void streamAPI(IndentedStringStream& ss) const {
    streamAPISignature(ss);

    ss.increaseIndent();
    ss << "use, intrinsic :: iso_c_binding" << endline;
    streamArgsDecls(ss);
  }

  void streamFooter(IndentedStringStream& ss) const {
    ss.decreaseIndent();
    bool isFunction = returnType_ != "";
    ss << "end " << (isFunction ? "function" : "subroutine") << endline;
  }

  std::string name_;
  std::string returnType_ = "";
  std::vector<std::tuple<std::string, std::string, int>> args_; // (name, type, dimensions)
  friend class FortranInterfaceModuleGen;
};

class FortranWrapperAPI : public FortranAPI {
public:
  FortranWrapperAPI(std::string name, std::optional<InterfaceType> returnType = std::nullopt)
      : FortranAPI(name, returnType) {}

  void addBodyLine(std::string line, bool withIndentation = true) {
    lines_.push_back(std::pair(line, withIndentation));
  }
  void addACCLine(std::string line) { lines_.push_back(std::pair("!$ACC " + line, false)); }

protected:
  void streamAPISignature(IndentedStringStream& ss) const override {
    FortranAPI::streamAPISignature(ss, /*isCBinding*/ false);
  }
  void streamArgsDecls(IndentedStringStream& ss) const override {
    std::for_each(args_.begin(), args_.end(), [&ss](auto& arg) {
      ss << std::get<1>(arg) << ", ";
      if(std::get<2>(arg) == 0) {
        ss << "value";
      } else {
        ss << "dimension(";
        {
          std::string sep;
          for(int c = 0; c < std::get<2>(arg); ++c) {
            ss << sep << ":";
            sep = ",";
          }
        }
        ss << ")";
      }
      ss << ", target :: " << std::get<0>(arg) << endline;
    });
  }

  void streamStatements(IndentedStringStream& ss) const {
    for(const auto& line : lines_) {
      if(line.second) { // with indentation
        ss << line.first << endline;
      } else {
        ss.ss() << line.first << std::endl;
      }
    }
  }

  // {(line, withIndentation)}
  std::vector<std::pair<std::string, bool>> lines_;
  friend class FortranInterfaceModuleGen;
};

class FortranInterfaceAPI : public FortranAPI {
public:
  FortranInterfaceAPI(std::string name, std::optional<InterfaceType> returnType = std::nullopt)
      : FortranAPI(name, returnType) {}

protected:
  void streamAPISignature(IndentedStringStream& ss) const override {
    FortranAPI::streamAPISignature(ss, /*isCBinding*/ true);
  }
  void streamArgsDecls(IndentedStringStream& ss) const override {
    std::for_each(args_.begin(), args_.end(), [&ss](auto& arg) {
      ss << std::get<1>(arg) << ", ";
      if(std::get<2>(arg) == 0) {
        ss << "value";
      } else {
        ss << "dimension(*)";
      }
      ss << ", target :: " << std::get<0>(arg) << endline;
    });
  }
};

class FortranInterfaceModuleGen {
public:
  FortranInterfaceModuleGen(IndentedStringStream& ss, std::string moduleName)
      : moduleName_(moduleName), ss_(ss) {}

  void addInterfaceAPI(const FortranInterfaceAPI& api) { interfaces_.push_back(api); }
  void addInterfaceAPI(FortranInterfaceAPI&& api) { interfaces_.push_back(std::move(api)); }

  void addWrapperAPI(const FortranWrapperAPI& api) { wrappers_.push_back(api); }
  void addWrapperAPI(FortranWrapperAPI&& api) { wrappers_.push_back(std::move(api)); }

  void commit() {
    ss_ << "module " << moduleName_ << endline;
    ss_ << "use, intrinsic :: iso_c_binding" << endline;
    ss_ << "implicit none" << endline;
    ss_.increaseIndent();
    ss_ << "interface" << endline;
    ss_.increaseIndent();

    for(auto& interface : interfaces_) {
      interface.streamAPI(ss_);
      interface.streamFooter(ss_);
    }

    ss_.decreaseIndent();
    ss_ << "end interface" << endline;
    if(!wrappers_.empty()) {
      ss_ << "contains" << endline;
      ss_.increaseIndent();
      for(auto& wrapper : wrappers_) {
        wrapper.streamAPI(ss_);
        wrapper.streamStatements(ss_);
        wrapper.streamFooter(ss_);
      }
      ss_.decreaseIndent();
    }

    ss_.decreaseIndent();
    ss_ << "end module" << endline;
  }

protected:
  std::string moduleName_;
  IndentedStringStream& ss_;
  std::vector<FortranInterfaceAPI> interfaces_;
  std::vector<FortranWrapperAPI> wrappers_;
};
} // namespace codegen
} // namespace dawn