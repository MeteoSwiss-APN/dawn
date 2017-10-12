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

#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/SIR/SIR.h"
#include "gtclang/Support/ParsingHeader.h"
#include "gtclang/Unittest/GTClang.h"
#include "gtclang/Unittest/UnittestEnvironment.h"
#include <fstream>
#include <gtest/gtest.h>
#include <map>

using namespace dawn;
using namespace gtclang;

namespace {

struct NestableFunctions : public codegen::MemberFunction {
  NestableFunctions(const Twine& type, const Twine& name, std::stringstream& s, int il = 0)
      : MemberFunction(type, name, s, il) {}

  NestableFunctions addFunction(const Twine& returntype, const Twine& name) {
    NestableFunctions nf(returntype, name, ss(), IndentLevel + 1);
    return nf;
  }

  NestableFunctions addVerticalRegion(const Twine& min, const Twine& max) {
    NestableFunctions nf(Twine::createNull(), "vertical_region", ss(), IndentLevel + 1);
    nf.addArg(min).addArg(max);
    nf.startBody();
    return nf;
  }

  NestableFunctions addIfStatement(const Twine& ifCondition) {
    NestableFunctions ifStatement(Twine::createNull(), "if", ss(), IndentLevel + 1);
    ifStatement.addArg(ifCondition);
    ifStatement.startBody();
    return ifStatement;
  }
};

struct VerticalRegion : public NestableFunctions {
  VerticalRegion(const Twine& name, std::stringstream& s, int il = 0)
      : NestableFunctions(Twine::createNull(), name, s, il) {}
};

struct DawnObject : public codegen::Structure {
  using Structure::Structure;
  DawnObject(const Twine& type, const Twine& name, std::stringstream& s)
      : Structure(type.str().c_str(), name, s) {}

  Statement addStorage(const Twine& memberName) {
    Statement member(ss(), IndentLevel + 1);
    member << "storage"
           << " " << memberName;
    return member;
  }

  NestableFunctions addDoMethod(const Twine& type) {
    NestableFunctions nf(type, "Do", ss(), IndentLevel + 1);
    return nf;
  }

  Statement addOffset(const Twine& offsetName) {
    Statement member(ss(), IndentLevel + 1);
    member << "offset"
           << " " << offsetName;
    return member;
  }
};

struct StencilFunction : public DawnObject {
  StencilFunction(const Twine& name, std::stringstream& s)
      : DawnObject("stencil_function", name, s) {}

  virtual NestableFunctions addDoMethod(const Twine& type) {
    NestableFunctions nf(type, "Do", ss(), IndentLevel + 1);
    nf.startBody();
    return nf;
  }
};

struct Stencil : public DawnObject {
  Stencil(const Twine& name, std::stringstream& s) : DawnObject("stencil", name, s) {}
};

struct Globals : public DawnObject {
  Globals(std::stringstream& s) : DawnObject("globals", Twine::createNull(), s) {}
  virtual NestableFunctions addDoMethod(const Twine& type) = delete;
  virtual Statement addOffset(const Twine& offsetName) = delete;
};

class FileWriter {
public:
  FileWriter(std::string filename) {
    filename_ = env_.getFileManager().dataPath() + filename;
    writeSkeleton();
  }

  void writeSkeleton() {
    fileHeader = HeaderWriter::longheader();
    fileHeader += HeaderWriter::includes();
  }

  std::stringstream& ss() { return ss_; }

  void writeToFile() {
    std::ofstream ofs(filename_, std::ofstream::trunc);
    ofs << fileHeader;
    ofs << ss_.str();
    ofs.close();
  }
  const std::string getFileName() const { return filename_; }
  UnittestEnvironment& getEnv() const { return env_; }

private:
  std::stringstream ss_;
  std::string filename_;
  std::string fileHeader;
  UnittestEnvironment& env_ = UnittestEnvironment::getSingleton();
};

TEST(ParsingTest, Assignment_Stencil) {
  FileWriter writer("AssignmentStencil.cpp");
  auto stencil = Stencil("test01", writer.ss());
  stencil.addStorage("a,b");

  auto domethod = stencil.addDoMethod("void");
  domethod.startBody();

  auto vr1 = domethod.addVerticalRegion("k_start", "k_end");
  vr1.addStatement("a = b");
  vr1.commit();
  domethod.commit();
  stencil.commit();
  writer.writeToFile();

  auto out =
      GTClang::run({writer.getFileName()}, writer.getEnv().getFlagManager().getDefaultFlags());

  ASSERT_TRUE(out.first);

  std::unique_ptr<dawn::SIR> test01SIR = make_unique<dawn::SIR>();
  test01SIR->Stencils.push_back(std::make_shared<dawn::sir::Stencil>());
  test01SIR->Stencils[0]->Name = "test01";
  test01SIR->Stencils[0]->StencilDescAst =
      std::make_shared<dawn::AST>(std::make_shared<BlockStmt>(std::vector<std::shared_ptr<Stmt>>{
          std::make_shared<VerticalRegionDeclStmt>(std::make_shared<sir::VerticalRegion>(
              std::make_shared<dawn::AST>(std::make_shared<BlockStmt>(
                  std::vector<std::shared_ptr<Stmt>>{std::make_shared<ExprStmt>(
                      std::make_shared<AssignmentExpr>(std::make_shared<FieldAccessExpr>("a"),
                                                       std::make_shared<FieldAccessExpr>("b")))})),
              std::make_shared<sir::Interval>(0, 1 << 20),
              sir::VerticalRegion::LoopOrderKind::LK_Forward))}));
  test01SIR->Stencils[0]->Fields.push_back(std::make_shared<sir::Field>("a"));
  test01SIR->Stencils[0]->Fields.push_back(std::make_shared<sir::Field>("b"));


  auto a = test01SIR->comparison(*out.second.get());
  EXPECT_TRUE(a.second) << a.first;
}
} // anonymous namespace
