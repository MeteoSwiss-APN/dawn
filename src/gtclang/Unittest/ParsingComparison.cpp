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
#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;
            )";
  }
};

struct NestableFunctions : public dawn::codegen::MemberFunction {
  NestableFunctions(const dawn::Twine& type, const dawn::Twine& name, std::stringstream& s,
                    int il = 0)
      : MemberFunction(type, name, s, il) {}

  NestableFunctions addFunction(const dawn::Twine& returntype, const dawn::Twine& name) {
    NestableFunctions nf(returntype, name, ss(), IndentLevel + 1);
    return nf;
  }

  NestableFunctions addVerticalRegion(const dawn::Twine& min, const dawn::Twine& max) {
    NestableFunctions nf(dawn::Twine::createNull(), "vertical_region", ss(), IndentLevel + 1);
    nf.addArg(min).addArg(max);
    nf.startBody();
    return nf;
  }

  NestableFunctions addIfStatement(const dawn::Twine& ifCondition) {
    NestableFunctions ifStatement(dawn::Twine::createNull(), "if", ss(), IndentLevel + 1);
    ifStatement.addArg(ifCondition);
    ifStatement.startBody();
    return ifStatement;
  }
};

struct VerticalRegion : public NestableFunctions {
  VerticalRegion(const dawn::Twine& name, std::stringstream& s, int il = 0)
      : NestableFunctions(dawn::Twine::createNull(), name, s, il) {}
};

struct DawnObject : public dawn::codegen::Structure {
  DawnObject(const dawn::Twine& type, const dawn::Twine& name, std::stringstream& s)
      : Structure(type.str().c_str(), name, s) {}

  Statement addStorage(const dawn::Twine& memberName) {
    Statement member(ss(), IndentLevel + 1);
    member << "storage"
           << " " << memberName;
    return member;
  }

  NestableFunctions addDoMethod(const dawn::Twine& type) {
    NestableFunctions nf(type, "Do", ss(), IndentLevel + 1);
    return nf;
  }

  Statement addOffset(const dawn::Twine& offsetName) {
    Statement member(ss(), IndentLevel + 1);
    member << "offset"
           << " " << offsetName;
    return member;
  }
};

struct StencilFunction : public DawnObject {
  StencilFunction(const dawn::Twine& name, std::stringstream& s)
      : DawnObject("stencil_function", name, s) {}

  virtual NestableFunctions addDoMethod(const dawn::Twine& type) {
    NestableFunctions nf(type, "Do", ss(), IndentLevel + 1);
    nf.startBody();
    return nf;
  }

  virtual ~StencilFunction() {}
};

struct Stencil : public DawnObject {
  Stencil(const dawn::Twine& name, std::stringstream& s) : DawnObject("stencil", name, s) {}
};

struct Globals : public DawnObject {
  Globals(std::stringstream& s) : DawnObject("globals", dawn::Twine::createNull(), s) {}
  virtual NestableFunctions addDoMethod(const dawn::Twine& type) = delete;
  virtual Statement addOffset(const dawn::Twine& offsetName) = delete;
  virtual ~Globals() {}
};

class FileWriter {
public:
  FileWriter(std::string testPath, std::string filename) {
    filename_ = UnittestEnvironment::getSingleton().getFileManager().dataPath() +
                "/gtclang/Frontend/" + testPath;
    auto errorRepport = llvm::sys::fs::create_directories(filename_);
    //    DAWN_ASSERT(errorRepport == llvm::instrprof_error::success);
    filename_ += filename;
    fileHeader_ = HeaderWriter::longheader();
    fileHeader_ += HeaderWriter::includes();
  }

  std::stringstream& ss() { return ss_; }

  void writeToFile() {
    std::ofstream ofs(filename_, std::ofstream::trunc);
    ofs << HeaderWriter::longheader();
    ofs << HeaderWriter::includes();
    ofs << ss_.str();
    ofs.close();
  }
  const std::string getFileName() const { return filename_; }

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
  std::string filename_;
  std::string fileHeader_;
};

CompareResult ParsingComparison::compare(const ParsedString& ps,
                                         const std::shared_ptr<dawn::Stmt>& stmt) {
  std::unique_ptr<dawn::SIR> test01SIR = dawn::make_unique<dawn::SIR>();
  wrapStatementInStencil(test01SIR, stmt);
  test01SIR->Filename = "In Memory Generated SIR";
  std::string localPath = UnittestEnvironment::getSingleton().testCaseName() + "/" +
                          UnittestEnvironment::getSingleton().testName() + "/";
  std::string fileName =
      dawn::format("TestStencil_%i.cpp", UnittestEnvironment::getSingleton().getUniqueID());
  FileWriter writer(localPath, fileName);
  writer.addParsedString(ps);
  auto out = GTClang::run({writer.getFileName()},
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
                                         const std::shared_ptr<dawn::Expr>& expr) {
  return compare(ps, std::make_shared<dawn::ExprStmt>(expr));
}

ParsingComparison* ParsingComparison::instance_ = nullptr;

ParsingComparison& ParsingComparison::getSingleton() {
  if(instance_ == nullptr)
    instance_ = new ParsingComparison;
  return *instance_;
}

void ParsingComparison::wrapStatementInStencil(std::unique_ptr<dawn::SIR>& sir,
                                               const std::shared_ptr<dawn::Stmt>& stmt) {
  using namespace dawn;
  if(BlockStmt* blockstmt = dawn::dyn_cast<BlockStmt>(stmt.get())) {
    sir->Stencils.push_back(std::make_shared<sir::Stencil>());
    sir->Stencils[0]->Name = "test01";
    sir->Stencils[0]->StencilDescAst =
        std::make_shared<AST>(std::make_shared<BlockStmt>(std::vector<std::shared_ptr<Stmt>>{
            std::make_shared<VerticalRegionDeclStmt>(std::make_shared<sir::VerticalRegion>(
                std::make_shared<AST>(std::make_shared<BlockStmt>(*blockstmt)),
                std::make_shared<sir::Interval>(sir::Interval::Start, sir::Interval::End),
                sir::VerticalRegion::LoopOrderKind::LK_Forward))}));
    auto allFields = dawn::getFieldFromStencilAST(sir->Stencils[0]->StencilDescAst);
    for(const auto& a : allFields) {
      sir->Stencils[0]->Fields.push_back(std::make_shared<dawn::sir::Field>(a));
    }
  } else {
    wrapStatementInStencil(sir,
                           std::make_shared<BlockStmt>(std::vector<std::shared_ptr<Stmt>>{stmt}));
  }
}

} // namespace gtclang
