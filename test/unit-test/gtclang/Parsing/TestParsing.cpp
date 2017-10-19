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
#include "gtclang/Unittest/SirGenerator.h"
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
    filename_ = UnittestEnvironment::getSingleton().getFileManager().dataPath() + filename;
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
  //  UnittestEnvironment& getEnv() const { return env_; }

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
  std::string fileHeader;
};

void wrapStatementInStencil(std::unique_ptr<dawn::SIR>& sir,
                            const std::shared_ptr<dawn::Stmt>& stmt) {
  using namespace dawn;
  if(BlockStmt* blockstmt = dawn::dyn_cast<BlockStmt>(stmt.get())) {
    sir->Stencils.push_back(std::make_shared<sir::Stencil>());
    sir->Stencils[0]->Name = "test01";
    sir->Stencils[0]->StencilDescAst =
        std::make_shared<AST>(std::make_shared<BlockStmt>(std::vector<std::shared_ptr<Stmt>>{
            std::make_shared<VerticalRegionDeclStmt>(std::make_shared<sir::VerticalRegion>(
                std::make_shared<AST>(std::make_shared<BlockStmt>(*blockstmt)),
                std::make_shared<sir::Interval>(0, sir::Interval::End),
                sir::VerticalRegion::LoopOrderKind::LK_Forward))}));
    FieldFinder ff;
    sir->Stencils[0]->StencilDescAst->accept(ff);
    for(const auto& a : ff.getFields()) {
      sir->Stencils[0]->Fields.push_back(a);
    }
  } else {
    wrapStatementInStencil(sir,
                           std::make_shared<BlockStmt>(std::vector<std::shared_ptr<Stmt>>{stmt}));
  }
}

std::pair<std::string, bool> compare(const ParsedString& ps,
                                     const std::shared_ptr<dawn::Stmt>& stmt) {
  std::unique_ptr<dawn::SIR> test01SIR = dawn::make_unique<dawn::SIR>();
  wrapStatementInStencil(test01SIR, stmt);
  test01SIR->Filename = "In Memory Generated SIR";
  FileWriter writer("TestStencil.cpp");
  writer.addParsedString(ps);
  auto out = GTClang::run({writer.getFileName()},
                          UnittestEnvironment::getSingleton().getFlagManager().getDefaultFlags());
  if(!out.first) {
    return std::make_pair("could not parse file " + writer.getFileName(), false);
  }
  auto compResult = test01SIR->comparison(*out.second.get());
  if(!compResult.second) {
    return std::make_pair(compResult.first, false);
  }
  return std::make_pair("", true);
}

std::pair<std::string, bool> compare(const ParsedString& ps,
                                     const std::shared_ptr<dawn::Expr>& expr) {
  return compare(ps, std::make_shared<dawn::ExprStmt>(expr));
}

#define DAWN_EXPECT_EQ(parsing, operation)                                                         \
  do {                                                                                             \
    auto output = compare(parsing, operation);                                                     \
    EXPECT_TRUE(output.second) << output.first;                                                    \
  } while(0)

#define DAWN_EXPECT_NE(parsing, operation)                                                         \
  do {                                                                                             \
    auto output = compare(parsing, operation);                                                     \
    EXPECT_FALSE(output.second) << "SIRs Match but should not";                                    \
  } while(0)

TEST(ParsingTest, Assignment) {
  // Field - Field
  DAWN_EXPECT_EQ(parse("a = b", field("a"), field("b")), assign(field("a"), field("b")));
  DAWN_EXPECT_EQ(parse("a += b", field("a"), field("b")), assign(field("a"), field("b"), "+="));
  DAWN_EXPECT_EQ(parse("a -= b", field("a"), field("b")), assign(field("a"), field("b"), "-="));
  DAWN_EXPECT_EQ(parse("a *= b", field("a"), field("b")), assign(field("a"), field("b"), "*="));
  DAWN_EXPECT_EQ(parse("a /= b", field("a"), field("b")), assign(field("a"), field("b"), "/="));

  //  Field - Variable
  DAWN_EXPECT_EQ(parse("float b = 1; a = b", field("a"), var("b")),
                 blockMultiple(vardec("float", "b", lit("1")), assign(field("a"), var("b"))));
  DAWN_EXPECT_EQ(parse("float b = 1; a += b", field("a"), var("b")),
                 blockMultiple(vardec("float", "b", lit("1")), assign(field("a"), var("b"), "+=")));
  DAWN_EXPECT_EQ(parse("float b = 1; a -= b", field("a"), var("b")),
                 blockMultiple(vardec("float", "b", lit("1")), assign(field("a"), var("b"), "-=")));
  DAWN_EXPECT_EQ(parse("float b = 1; a *= b", field("a"), var("b")),
                 blockMultiple(vardec("float", "b", lit("1")), assign(field("a"), var("b"), "*=")));
  DAWN_EXPECT_EQ(parse("float b = 1; a /= b", field("a"), var("b")),
                 blockMultiple(vardec("float", "b", lit("1")), assign(field("a"), var("b"), "/=")));

  // Variable - Field
  DAWN_EXPECT_EQ(parse("float b = 1; b = a", field("a"), var("b")),
                 blockMultiple(vardec("float", "b", lit("1")), assign(var("b"), field("a"))));
  DAWN_EXPECT_EQ(parse("float b = 1; b += a", field("a"), var("b")),
                 blockMultiple(vardec("float", "b", lit("1")), assign(var("b"), field("a"), "+=")));
  DAWN_EXPECT_EQ(parse("float b = 1; b -= a", field("a"), var("b")),
                 blockMultiple(vardec("float", "b", lit("1")), assign(var("b"), field("a"), "-=")));
  DAWN_EXPECT_EQ(parse("float b = 1; b *= a", field("a"), var("b")),
                 blockMultiple(vardec("float", "b", lit("1")), assign(var("b"), field("a"), "*=")));
  DAWN_EXPECT_EQ(parse("float b = 1; b /= a", field("a"), var("b")),
                 blockMultiple(vardec("float", "b", lit("1")), assign(var("b"), field("a"), "/=")));

  // Variable - Variable
  DAWN_EXPECT_EQ(
      parse("float b = 1; float c = 2; b = c; field = b", field("field"), var("b"), var("c")),
      blockMultiple(vardec("float", "b", lit("1")), vardec("float", "c", lit("2")),
                    assign(var("b"), var("c")), assign(field("field"), var("b"))));
  DAWN_EXPECT_EQ(
      parse("float b = 1; float c = 2; b += c; field = b", field("field"), var("b"), var("c")),
      blockMultiple(vardec("float", "b", lit("1")), vardec("float", "c", lit("2")),
                    assign(var("b"), var("c"), "+="), assign(field("field"), var("b"))));
  DAWN_EXPECT_EQ(
      parse("float b = 1; float c = 2; b -= c; field = b", field("field"), var("b"), var("c")),
      blockMultiple(vardec("float", "b", lit("1")), vardec("float", "c", lit("2")),
                    assign(var("b"), var("c"), "-="), assign(field("field"), var("b"))));
  DAWN_EXPECT_EQ(
      parse("float b = 1; float c = 2; b *= c; field = b", field("field"), var("b"), var("c")),
      blockMultiple(vardec("float", "b", lit("1")), vardec("float", "c", lit("2")),
                    assign(var("b"), var("c"), "*="), assign(field("field"), var("b"))));
  DAWN_EXPECT_EQ(
      parse("float b = 1; float c = 2; b /= c; field = b", field("field"), var("b"), var("c")),
      blockMultiple(vardec("float", "b", lit("1")), vardec("float", "c", lit("2")),
                    assign(var("b"), var("c"), "/="), assign(field("field"), var("b"))));
}

TEST(ParsingTest, Unop) {
  DAWN_EXPECT_EQ(parse(R"(
                       float a = 0;
                       ++a;
                       st = a;)",
                       field("st")),
                 blockMultiple(vardec("float", "a", lit("0")), unop(var("a"), "++"),
                               assign(field("st"), var("a"))));
  DAWN_EXPECT_EQ(parse(R"(
                       float a = 0;
                       a--;
                       st = a;)",
                       field("st")),
                 blockMultiple(vardec("float", "a", lit("0")), unop(var("a"), "--"),
                               assign(field("st"), var("a"))));
}

TEST(ParsingTest, BinOp) {
  DAWN_EXPECT_EQ(parse("a = b + c", field("a"), field("b"), field("c")),
                 assign(field("a"), binop(field("b"), "+", field("c"))));
  DAWN_EXPECT_EQ(parse("a = b - c", field("a"), field("b"), field("c")),
                 assign(field("a"), binop(field("b"), "-", field("c"))));
  DAWN_EXPECT_EQ(parse("a = b / c", field("a"), field("b"), field("c")),
                 assign(field("a"), binop(field("b"), "/", field("c"))));
  DAWN_EXPECT_EQ(parse("a = b * c", field("a"), field("b"), field("c")),
                 assign(field("a"), binop(field("b"), "*", field("c"))));

  DAWN_EXPECT_EQ(parse("a = b < c", field("a"), field("b"), field("c")),
                 assign(field("a"), binop(field("b"), "<", field("c"))));
  DAWN_EXPECT_EQ(parse("a = b > c", field("a"), field("b"), field("c")),
                 assign(field("a"), binop(field("b"), ">", field("c"))));
  DAWN_EXPECT_EQ(parse("a = b == c", field("a"), field("b"), field("c")),
                 assign(field("a"), binop(field("b"), "==", field("c"))));
  DAWN_EXPECT_EQ(parse("a = b >= c", field("a"), field("b"), field("c")),
                 assign(field("a"), binop(field("b"), ">=", field("c"))));
  DAWN_EXPECT_EQ(parse("a = b <= c", field("a"), field("b"), field("c")),
                 assign(field("a"), binop(field("b"), "<=", field("c"))));

  DAWN_EXPECT_EQ(parse("a = b && c", field("a"), field("b"), field("c")),
                 assign(field("a"), binop(field("b"), "&&", field("c"))));
  DAWN_EXPECT_EQ(parse("a = b || c", field("a"), field("b"), field("c")),
                 assign(field("a"), binop(field("b"), "||", field("c"))));

  DAWN_EXPECT_EQ(parse(R"(
                       int b = 1;
                       int c = 2;
                       a = b & c;)",
                       field("a"), var("b"), var("c")),
                 blockMultiple(vardec("int", "b", lit("1")), vardec("int", "c", lit("2")),
                               assign(field("a"), binop(var("b"), "&", var("c")))));
  DAWN_EXPECT_EQ(parse(R"(
                       int b = 1;
                       int c = 2;
                       a = b | c;)",
                       field("a"), var("b"), var("c")),
                 blockMultiple(vardec("int", "b", lit("1")), vardec("int", "c", lit("2")),
                               assign(field("a"), binop(var("b"), "|", var("c")))));
  DAWN_EXPECT_EQ(parse(R"(
                       int b = 1;
                       int c = 2;
                       a = b ^ c;)",
                       field("a"), var("b"), var("c")),
                 blockMultiple(vardec("int", "b", lit("1")), vardec("int", "c", lit("2")),
                               assign(field("a"), binop(var("b"), "^", var("c")))));
  DAWN_EXPECT_EQ(parse(R"(
                        int b = 1;
                        int c = 2;
                        a = b << c;)",
                       field("a"), var("b"), var("c")),
                 blockMultiple(vardec("int", "b", lit("1")), vardec("int", "c", lit("2")),
                               assign(field("a"), binop(var("b"), "<<", var("c")))));
  DAWN_EXPECT_EQ(parse(R"(
                       int b = 1;
                       int c = 2;
                       a = b >> c
                       )",
                       field("a"), var("b"), var("c")),
                 blockMultiple(vardec("int", "b", lit("1")), vardec("int", "c", lit("2")),
                               assign(field("a"), binop(var("b"), ">>", var("c")))));
}

TEST(ParsingTest, TernOp) {
  DAWN_EXPECT_EQ(parse(R"(
                       float a = 0;
                       float b = 1;
                       float c = 2;
                       f = a<0 ? b : c;)",
                       field("f"), var("a"), var("b"), var("c")),
                 blockMultiple(vardec("float", "a", lit("0")), vardec("float", "b", lit("1")),
                               vardec("float", "c", lit("2")),
                               assign(field("f"),
                                      ternop(binop(var("a"), "<", lit("0")), var("b"), var("c")))));
}

TEST(ParsingTest, IfStmt) {
  DAWN_EXPECT_EQ(parse(R"(
                         float a = 0;
                         float b = 1;
                         if(b > 0){
                         fieldOne = a;
                         } else {
                         fieldTwo = b;
                         })",
                       field("fieldOne"), field("fieldTwo")),
                 blockMultiple(vardec("float", "a", lit("0")), vardec("float", "b", lit("1")),
                               ifst(expr(binop(var("b"), ">", lit("0"))),
                                    blockMultiple(expr(assign(field("fieldOne"), var("a")))),
                                    blockMultiple(expr(assign(field("fieldTwo"), var("b")))))));
  DAWN_EXPECT_EQ(parse(R"(
                      float a = 0;
                      float b = 1;
                      if(b > 0){
                      field = a;
                      })",
                       field("field")),
                 blockMultiple(vardec("float", "a", lit("0")), vardec("float", "b", lit("1")),
                               ifst(expr(binop(var("b"), ">", lit("0"))),
                                    blockMultiple(expr(assign(field("field"), var("a")))))));
}

#undef DAWN_EXPECT_EQ

#undef DAWN_EXPECT_NE

} // anonymous namespace
