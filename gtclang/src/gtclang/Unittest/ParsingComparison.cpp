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

#include "gtclang/Unittest/ParsingComparison.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/SIR/ASTUtil.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Array.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/Unreachable.h"
#include "gtclang/Unittest/GTClang.h"
#include "gtclang/Unittest/UnittestEnvironment.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <fstream>

namespace gtclang {

struct HeaderWriter {
  HeaderWriter() = delete;
  static std::string longheader() {
    return R"(
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
           )";
  }
  static std::string includes() {
    return R"(
#include "gtclang_dsl_defs/gtclang_dsl.hpp"

using namespace gtclang::dsl;
            )";
  }
};

struct NestableFunctions : public dawn::codegen::MemberFunction {
  NestableFunctions(const std::string& type, const std::string& name, std::stringstream& s,
                    int il = 0)
      : MemberFunction(type, name, s, il) {}

  NestableFunctions addFunction(const std::string& returntype, const std::string& name) {
    NestableFunctions nf(returntype, name, ss(), IndentLevel + 1);
    return nf;
  }

  NestableFunctions addVerticalRegion(const std::string& min, const std::string& max) {
    NestableFunctions nf("", "vertical_region", ss(), IndentLevel + 1);
    nf.addArg(min).addArg(max);
    nf.startBody();
    return nf;
  }

  NestableFunctions addIfStatement(const std::string& ifCondition) {
    NestableFunctions ifStatement("", "if", ss(), IndentLevel + 1);
    ifStatement.addArg(ifCondition);
    ifStatement.startBody();
    return ifStatement;
  }
};

struct VerticalRegion : public NestableFunctions {
  VerticalRegion(const std::string& name, std::stringstream& s, int il = 0)
      : NestableFunctions("", name, s, il) {}
};

struct StencilBase : public dawn::codegen::Structure {
  StencilBase(const std::string& type, const std::string& name, std::stringstream& s)
      : Structure(type.c_str(), name, s) {}

  Statement addStorage(const std::string& memberName) {
    Statement member(ss(), IndentLevel + 1);
    member << "storage"
           << " " << memberName;
    return member;
  }

  NestableFunctions addDoMethod(const std::string& type) {
    NestableFunctions nf(type, "Do", ss(), IndentLevel + 1);
    return nf;
  }

  Statement addOffset(const std::string& offsetName) {
    Statement member(ss(), IndentLevel + 1);
    member << "offset"
           << " " << offsetName;
    return member;
  }
};

struct StencilFunction : public StencilBase {
  StencilFunction(const std::string& name, std::stringstream& s)
      : StencilBase("stencil_function", name, s) {}

  virtual NestableFunctions addDoMethod(const std::string& type) {
    NestableFunctions nf(type, "Do", ss(), IndentLevel + 1);
    nf.startBody();
    return nf;
  }

  virtual ~StencilFunction() {}
};

struct Stencil : public StencilBase {
  Stencil(const std::string& name, std::stringstream& s) : StencilBase("stencil", name, s) {}
};

struct Globals : public StencilBase {
  Globals(std::stringstream& s) : StencilBase("globals", "", s) {}
  virtual NestableFunctions addDoMethod(const std::string& type) = delete;
  virtual Statement addOffset(const std::string& offsetName) = delete;
  virtual ~Globals() {}
};

class FileWriter {
public:
  FileWriter(std::string localPath, std::string filename)
      : localPath_(localPath), filename_(filename) {
    fullpath_ =
        UnittestEnvironment::getSingleton().getFileManager().getUnittestFile(localPath, filename);
    fileHeader_ = HeaderWriter::longheader();
    fileHeader_ += HeaderWriter::includes();
  }

  std::stringstream& ss() { return ss_; }

  void writeToFile() {
    std::ofstream ofs(fullpath_, std::ofstream::trunc);
    ofs << HeaderWriter::longheader();
    ofs << HeaderWriter::includes();
    ofs << ss_.str();
    ofs.close();
  }
  const std::string getFileName() const { return fullpath_; }

  void addParsedString(const ParsedString& ps) {
    auto stencil = Stencil("test01", ss_);
    for(const auto& a : ps.getFields()) {
      stencil.addStorage(a);
    }
    auto doMethod = stencil.addDoMethod("void");
    doMethod.startBody();

    auto vr = doMethod.addVerticalRegion("k_start", "k_end");
    vr.addStatement(ps.getCall());
    vr.commit();
    doMethod.commit();
    stencil.commit();
    writeToFile();
  }

private:
  std::stringstream ss_;
  std::string fullpath_;
  std::string fileHeader_;
  std::string localPath_;
  std::string filename_;
};

class FieldFinder : public dawn::ast::ASTVisitorForwarding {
public:
  virtual void visit(const std::shared_ptr<dawn::ast::FieldAccessExpr>& expr) {
    auto fieldFromExpression = dawn::sir::Field(
        expr->getName(),
        dawn::sir::FieldDimensions(
            dawn::sir::HorizontalFieldDimension(dawn::ast::cartesian, {true, true}), true));

    auto iter = std::find(allFields_.begin(), allFields_.end(), fieldFromExpression);
    if(iter == allFields_.end())
      allFields_.push_back(fieldFromExpression);
    dawn::ast::ASTVisitorForwarding::visit(expr);
  }

  virtual void visit(const std::shared_ptr<dawn::sir::VerticalRegionDeclStmt>& stmt) {
    stmt->getVerticalRegion()->Ast->accept(*this);
  }

  const std::vector<dawn::sir::Field>& getFields() const { return allFields_; }

private:
  std::vector<dawn::sir::Field> allFields_;
};

extern std::vector<dawn::sir::Field>
getFieldFromStencilAST(const std::shared_ptr<dawn::ast::AST>& ast) {
  FieldFinder finder;
  ast->accept(finder);
  return finder.getFields();
}

CompareResult ParsingComparison::compare(const ParsedString& ps,
                                         const std::shared_ptr<dawn::sir::Stmt>& stmt) {
  std::unique_ptr<dawn::SIR> test01SIR =
      std::make_unique<dawn::SIR>(dawn::ast::GridType::Cartesian);
  wrapStatementInStencil(test01SIR, stmt);
  test01SIR->Filename = "In Memory Generated SIR";
  std::string localPath = "Frontend/" + UnittestEnvironment::getSingleton().testCaseName() + "/" +
                          UnittestEnvironment::getSingleton().testName();
  std::string fileName =
      dawn::format("TestStencil_%i.cpp", UnittestEnvironment::getSingleton().getUniqueID());
  FileWriter writer(localPath, fileName);
  writer.addParsedString(ps);
  std::pair<bool, std::shared_ptr<dawn::SIR>> out =
      GTClang::run({writer.getFileName(), "-fno-codegen"},
                   UnittestEnvironment::getSingleton().getFlagManager().getDefaultFlags());
  if(!out.first) {
    return CompareResult{"could not parse file " + writer.getFileName(), false};
  }
  auto compResult = test01SIR->comparison(*out.second.get());
  if(!bool(compResult)) {
    return CompareResult{compResult.why(), false};
  }
  return CompareResult{"", true};
}

CompareResult ParsingComparison::compare(const ParsedString& ps,
                                         const std::shared_ptr<dawn::sir::Expr>& expr) {
  return compare(ps, dawn::sir::makeExprStmt(expr));
}

ParsingComparison* ParsingComparison::instance_ = nullptr;

ParsingComparison& ParsingComparison::getSingleton() {
  if(instance_ == nullptr)
    instance_ = new ParsingComparison;
  return *instance_;
}

void ParsingComparison::wrapStatementInStencil(std::unique_ptr<dawn::SIR>& sir,
                                               const std::shared_ptr<dawn::sir::Stmt>& stmt) {
  using namespace dawn;
  if(sir::BlockStmt* blockstmt = dawn::dyn_cast<sir::BlockStmt>(stmt.get())) {
    sir->Stencils.push_back(std::make_shared<sir::Stencil>());
    sir->Stencils[0]->Name = "test01";
    sir->Stencils[0]->StencilDescAst =
        std::make_shared<sir::AST>(sir::makeBlockStmt(std::vector<std::shared_ptr<sir::Stmt>>{
            sir::makeVerticalRegionDeclStmt(std::make_shared<sir::VerticalRegion>(
                std::make_shared<sir::AST>(std::make_shared<sir::BlockStmt>(*blockstmt)),
                std::make_shared<sir::Interval>(sir::Interval::Start, sir::Interval::End),
                sir::VerticalRegion::LoopOrderKind::Forward))}));
    auto allFields = getFieldFromStencilAST(sir->Stencils[0]->StencilDescAst);
    for(const auto& a : allFields) {
      sir->Stencils[0]->Fields.push_back(std::make_shared<dawn::sir::Field>(a));
    }
  } else {
    wrapStatementInStencil(sir, sir::makeBlockStmt(std::vector<std::shared_ptr<sir::Stmt>>{stmt}));
  }
}

} // namespace gtclang
