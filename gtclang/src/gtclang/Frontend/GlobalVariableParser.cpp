//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _
//                        | |     | |
//                    __ _| |_ ___| | __ _ _ __   __ _
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gtclang/Frontend/GlobalVariableParser.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Unreachable.h"
#include "gtclang/Frontend/GTClangContext.h"
#include "gtclang/Support/ASTUtils.h"
#include "gtclang/Support/ClangCompat/SourceLocation.h"
#include "gtclang/Support/FileUtil.h"
#include "gtclang/Support/Logger.h"
#include "clang/AST/AST.h"
#include <cstdlib>
#include <fstream>

namespace gtclang {

namespace {

//===------------------------------------------------------------------------------------------===//
//     StringInitArgResolver
//===------------------------------------------------------------------------------------------===//

/// @brief Extracts the `RHS` of an in-class initialization of `std::string str = "RHS"`
class StringInitArgResolver {
  std::string str_;

public:
  void resolve(clang::Expr* expr) {
    using namespace clang;
    // ignore implicit nodes
    expr = skipAllImplicitNodes(expr);

    if(CXXDefaultArgExpr* e = dyn_cast<CXXDefaultArgExpr>(expr))
      resolve(e);
    else if(CXXConstructExpr* e = dyn_cast<CXXConstructExpr>(expr))
      resolve(e);
    else if(StringLiteral* e = dyn_cast<StringLiteral>(expr))
      resolve(e);
    else {
      DAWN_ASSERT_MSG(0, "unresolved expression in StringInitArgResolver");
      llvm_unreachable("invalid expr");
    }
  }

  void resolve(clang::CXXDefaultArgExpr* expr) { resolve(expr->getExpr()); }

  void resolve(clang::CXXConstructExpr* expr) {
    for(int i = 0; i < expr->getNumArgs(); ++i)
      resolve(expr->getArg(i));
  }

  void resolve(clang::StringLiteral* expr) { str_ = expr->getString().str(); }

  /// @brief Get the parsed string (may be empty)
  std::string getStr() { return str_; }
};

} // anonymous namespace

//===------------------------------------------------------------------------------------------===//
//     GlobalVariableParser
//===------------------------------------------------------------------------------------------===//

GlobalVariableParser::GlobalVariableParser(gtclang::GTClangContext* context)
    : context_(context), variableMap_(std::make_shared<dawn::sir::GlobalVariableMap>()),
      configFile_(std::make_shared<dawn::json::json>()), recordDecl_(nullptr) {}

const std::shared_ptr<dawn::sir::GlobalVariableMap>&
GlobalVariableParser::getGlobalVariableMap() const {
  return variableMap_;
}

bool GlobalVariableParser::has(const std::string& name) const { return variableMap_->count(name); }

void GlobalVariableParser::parseGlobals(clang::CXXRecordDecl* recordDecl) {
  using namespace clang;

  variableMap_->clear();
  configFile_->clear();

  DAWN_LOG(INFO)
      << "Parsing globals at "
      << getFilename(recordDecl->getLocation().printToString(context_->getSourceManager())).str()
      << " ...";

  recordDecl_ = recordDecl;

  for(FieldDecl* arg : recordDecl->fields()) {
    auto name = arg->getDeclName().getAsString();

    DAWN_LOG(INFO) << "Parsing global variable: " << name;

    // Extract the type `T` of the field `T var;`
    auto type = arg->getType();
    DAWN_ASSERT(!type.isNull());

    dawn::sir::Value::TypeKind typeKind;
    if(type->isBooleanType()) // bool
      typeKind = dawn::sir::Value::Boolean;
    else if(type->isIntegerType()) // int
      typeKind = dawn::sir::Value::Integer;
    else if(type->isArithmeticType()) // int, float, double... we treat this as 'double'
      typeKind = dawn::sir::Value::Double;
    else if(type.getAsString() == "std::string") {
      typeKind = dawn::sir::Value::String;
    } else {
      context_->getDiagnostics().report(arg->getLocation(), Diagnostics::err_globals_invalid_type)
          << type.getAsString() << name;
      context_->getDiagnostics().report(arg->getLocation(),
                                        Diagnostics::note_globals_allowed_types);
      return;
    }

    std::shared_ptr<dawn::sir::Value> value;

    // Check if we have a default value `value` i.e `T var = value`
    if(arg->hasInClassInitializer()) {
      Expr* init = skipAllImplicitNodes(arg->getInClassInitializer());

      auto reportError = [&]() {
        context_->getDiagnostics().report(clang_compat::getBeginLoc(*init),
                                          Diagnostics::err_globals_invalid_default_value)
            << init->getType().getAsString() << name;
      };

      // demotion to integer (`double ->12<-' would dyncast to int)
      if(dyn_cast<IntegerLiteral>(init) != nullptr && typeKind == dawn::sir::Value::Integer) {
        IntegerLiteral* il = dyn_cast<IntegerLiteral>(init);
        std::string valueStr = il->getValue().toString(10, true);
        value = std::make_shared<dawn::sir::Value>((int)std::atoi(valueStr.c_str()));
        DAWN_LOG(INFO) << "Setting default value of '" << name << "' to '" << valueStr << "'";

      // this slightly unelegant procedure is needed since FloatingLiteral does not cast from
      // expressions without trailing do (e.g. `12.' would cast, `12' wouldn't.)
      } else if((dyn_cast<FloatingLiteral>(init) != nullptr || dyn_cast<IntegerLiteral>(init) != nullptr)
          && typeKind == dawn::sir::Value::Double) {
        IntegerLiteral* il = dyn_cast<IntegerLiteral>(init);
        FloatingLiteral* fl = dyn_cast<FloatingLiteral>(init);
        std::string valueStr;
        if (fl != nullptr) {
          llvm::SmallVector<char, 10> valueVec;
          fl->getValue().toString(valueVec);
          valueStr = std::string(valueVec.data(), valueVec.size());
          value = std::make_shared<dawn::sir::Value>( (double) std::atof(valueStr.c_str() ));
        } else {
          valueStr = il->getValue().toString(10, true);
          value = std::make_shared<dawn::sir::Value>( (double) std::atof(valueStr.c_str() ));
        }
        DAWN_LOG(INFO) << "Setting default value of '" << name << "' to '" << valueStr << "'";

      } else if(CXXBoolLiteralExpr* bl = dyn_cast<CXXBoolLiteralExpr>(init)) {
         value = std::make_shared<dawn::sir::Value>((bool) bl->getValue());
        DAWN_LOG(INFO) << "Setting default value of '" << name << "' to '" << bl->getValue() << "'";

      } else if(typeKind == dawn::sir::Value::String) {
        StringInitArgResolver resolver;
        resolver.resolve(init);
        std::string valueStr = resolver.getStr();

        if(!valueStr.empty()) {
          value = std::make_shared<dawn::sir::Value>(valueStr);
          DAWN_LOG(INFO) << "Setting default value of '" << name << "' to '" << valueStr << "'";
        } else
          reportError();

      } else {
        reportError();
      }
    } else {
      value = std::make_shared<dawn::sir::Value>(typeKind);
    }

    variableMap_->emplace(name, value);
  }

  // Try to parse the config file
  if(!context_->getOptions().ConfigFile.empty()) {
    std::string& file = context_->getOptions().ConfigFile;

    DAWN_LOG(INFO) << "Reading global config file \"" << file << "\"";

    std::ifstream fin(file);
    if(!fin.is_open()) {
      context_->getDiagnostics().report(Diagnostics::err_fs_error)
          << dawn::format("cannot to open config file '%s'", file);
      return;
    }

    auto configError = [&](const std::string& msg) {
      context_->getDiagnostics().report(Diagnostics::err_globals_config)
          << ("invalid json: " + msg);
    };

    try {
      fin >> *configFile_;
      fin.close();
    } catch(std::exception& e) {
      configError(e.what());
      return;
    }

    if(!configFile_->count("globals")) {
      configError("expected top-level key \"globals\"");
      return;
    }

    const auto& globals = (*configFile_)["globals"];

    for(auto it = globals.begin(), end = globals.end(); it != end; ++it) {
      std::string key = it.key();

      auto varIt = variableMap_->find(key);
      if(varIt == variableMap_->end()) {
        DAWN_LOG(INFO) << "Parse non-existing variable: " << key;
        continue;
      }

      dawn::sir::Value& value = *varIt->second;
      std::shared_ptr<dawn::sir::Value> parsed_value;
      
      // Treat the value as a compile time constant
      //  i.e., at this point in time we are sure that this is a compile time constant
      const bool isConstExpr = true;

      try {
        switch(value.getType()) {
        case dawn::sir::Value::Boolean:
          parsed_value = std::make_shared<dawn::sir::Value>(bool(*it), isConstExpr);
          break;
        case dawn::sir::Value::Integer:
          parsed_value = std::make_shared<dawn::sir::Value>(int(*it), isConstExpr);
          break;
        case dawn::sir::Value::Double:
          parsed_value = std::make_shared<dawn::sir::Value>(double(*it), isConstExpr);
          break;
        case dawn::sir::Value::String: {
          std::string v = *it;
          parsed_value = std::make_shared<dawn::sir::Value>(v, isConstExpr);
          }
          break;
        default:
          dawn_unreachable("invalid type");
        }
      } catch(std::domain_error& e) {
        configError("invalid key '" + key + "': " + e.what());
        return;
      }

      variableMap_->at(key) = parsed_value;  //update varIt in map
      assert(variableMap_->at(key)->has_value());

      DAWN_LOG(INFO) << "Setting constant value of '" << key << " to '" << variableMap_->at(key)->toString() << "'";     
    }
  }

  DAWN_LOG(INFO) << "Done parsing globals";
}

clang::CXXRecordDecl* GlobalVariableParser::getRecordDecl() const { return recordDecl_; }

} // namespace gtclang
