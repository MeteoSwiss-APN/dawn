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
#include "dawn/Support/FileSystem.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Unreachable.h"
#include "gtclang/Frontend/GTClangContext.h"
#include "gtclang/Support/ASTUtils.h"
#include "gtclang/Support/ClangCompat/SourceLocation.h"
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
    : context_(context), variableMap_(std::make_shared<dawn::ast::GlobalVariableMap>()),
      configFile_(std::make_shared<dawn::json::json>()), recordDecl_(nullptr) {}

const std::shared_ptr<dawn::ast::GlobalVariableMap>&
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
      << fs::path(recordDecl->getLocation().printToString(context_->getSourceManager())).filename()
      << " ...";

  recordDecl_ = recordDecl;

  for(FieldDecl* arg : recordDecl->fields()) {
    auto name = arg->getDeclName().getAsString();

    DAWN_LOG(INFO) << "Parsing global variable: " << name;

    // Extract the type `T` of the field `T var;`
    auto type = arg->getType();
    DAWN_ASSERT(!type.isNull());

    dawn::ast::Value::Kind typeKind;
    if(type->isBooleanType()) // bool
      typeKind = dawn::ast::Value::Kind::Boolean;
    else if(type->isIntegerType()) // int
      typeKind = dawn::ast::Value::Kind::Integer;
    else if(type->isArithmeticType()) // int, float, double... we treat this as 'double'
      typeKind = dawn::ast::Value::Kind::Double;
    else if(type.getAsString() == "std::string") {
      typeKind = dawn::ast::Value::Kind::String;
    } else {
      context_->getDiagnostics().report(arg->getLocation(), Diagnostics::err_globals_invalid_type)
          << type.getAsString() << name;
      context_->getDiagnostics().report(arg->getLocation(),
                                        Diagnostics::note_globals_allowed_types);
      return;
    }

    std::shared_ptr<dawn::ast::Global> value = 0;

    // Check if we have a default value `value` i.e `T var = value`
    if(arg->hasInClassInitializer()) {
      Expr* init = skipAllImplicitNodes(arg->getInClassInitializer());

      auto reportError = [&]() {
        context_->getDiagnostics().report(clang_compat::getBeginLoc(*init),
                                          Diagnostics::err_globals_invalid_default_value)
            << init->getType().getAsString() << name;
      };

      // demotion to integer (`double ->12<-' would dyncast to int)
      if(dyn_cast<IntegerLiteral>(init) != nullptr && typeKind == dawn::ast::Value::Kind::Integer) {
        IntegerLiteral* il = dyn_cast<IntegerLiteral>(init);
        std::string valueStr = il->getValue().toString(10, true);
        value = std::make_shared<dawn::ast::Global>((int)std::atoi(valueStr.c_str()));
        DAWN_LOG(INFO) << "Setting default value of '" << name << "' to '" << valueStr << "'";

        // this slightly unelegant procedure is needed since FloatingLiteral does not cast from
        // expressions without trailing do (e.g. `12.' would cast, `12' wouldn't.)
      } else if((dyn_cast<FloatingLiteral>(init) != nullptr ||
                 dyn_cast<IntegerLiteral>(init) != nullptr) &&
                typeKind == dawn::ast::Value::Kind::Double) {
        IntegerLiteral* il = dyn_cast<IntegerLiteral>(init);
        FloatingLiteral* fl = dyn_cast<FloatingLiteral>(init);
        std::string valueStr;
        if(fl != nullptr) {
          llvm::SmallVector<char, 10> valueVec;
          fl->getValue().toString(valueVec);
          valueStr = std::string(valueVec.data(), valueVec.size());
          value = std::make_shared<dawn::ast::Global>((double)std::atof(valueStr.c_str()));
        } else {
          valueStr = il->getValue().toString(10, true);
          value = std::make_shared<dawn::ast::Global>((double)std::atof(valueStr.c_str()));
        }
        DAWN_LOG(INFO) << "Setting default value of '" << name << "' to '" << valueStr << "'";

      } else if(CXXBoolLiteralExpr* bl = dyn_cast<CXXBoolLiteralExpr>(init)) {
        value = std::make_shared<dawn::ast::Global>((bool)bl->getValue());
        DAWN_LOG(INFO) << "Setting default value of '" << name << "' to '" << bl->getValue() << "'";

      } else if(typeKind == dawn::ast::Value::Kind::String) {
        StringInitArgResolver resolver;
        resolver.resolve(init);
        std::string valueStr = resolver.getStr();

        if(!valueStr.empty()) {
          value = std::make_shared<dawn::ast::Global>(valueStr);
          DAWN_LOG(INFO) << "Setting default value of '" << name << "' to '" << valueStr << "'";
        } else
          reportError();

      } else {
        reportError();
      }
    } else {
      value = std::make_shared<dawn::ast::Global>(typeKind);
    }

    if(value) {
      variableMap_->emplace(std::pair(std::string(name), std::move(*value)));
    }
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

      dawn::ast::Global& global = varIt->second;
      std::shared_ptr<dawn::ast::Global> parsed_global;

      // Treat the value as a compile time constant
      //  i.e., at this point in time we are sure that this is a compile time constant
      const bool isConstExpr = true;

      try {
        switch(global.getType()) {
        case dawn::ast::Value::Kind::Boolean:
          parsed_global = std::make_shared<dawn::ast::Global>(bool(*it), isConstExpr);
          break;
        case dawn::ast::Value::Kind::Integer:
          parsed_global = std::make_shared<dawn::ast::Global>(int(*it), isConstExpr);
          break;
        case dawn::ast::Value::Kind::Double:
          parsed_global = std::make_shared<dawn::ast::Global>(double(*it), isConstExpr);
          break;
        case dawn::ast::Value::Kind::String: {
          std::string v = *it;
          parsed_global = std::make_shared<dawn::ast::Global>(v, isConstExpr);
        } break;
        default:
          dawn_unreachable("invalid type");
        }
      } catch(std::domain_error& e) {
        configError("invalid key '" + key + "': " + e.what());
        return;
      }

      // update varIt in map
      variableMap_->erase(key);
      variableMap_->insert(std::pair(std::string(key), std::move(*parsed_global)));

      DAWN_LOG(INFO) << "Setting constant value of '" << key << " to '"
                     << variableMap_->at(key).toString() << "'";
    }
  }

  DAWN_LOG(INFO) << "Done parsing globals";
}

clang::CXXRecordDecl* GlobalVariableParser::getRecordDecl() const { return recordDecl_; }

} // namespace gtclang
