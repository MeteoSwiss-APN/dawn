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
#include "gtclang/Unittest/UnittestStmtSimplifier.h"
#include <fstream>
#include <gtest/gtest.h>

using namespace gtclang;
using namespace sirgen;

namespace {

#define DAWN_EXPECT_EQ(parsing, operation)                                                         \
  do {                                                                                             \
    auto output = ParsingComparison::getSingleton().compare(parsing, operation);                   \
    EXPECT_TRUE(bool(output)) << output.why();                                                     \
  } while(0)

#define DAWN_EXPECT_NE(parsing, operation)                                                         \
  do {                                                                                             \
    auto output = ParsingComparison::getSingleton().compare(parsing, operation);                   \
    EXPECT_FALSE(static_cast<bool>(output)) << "SIRs Match but should not"; \
  } while(0)

TEST(ParsingTest, Setup) {
  // Setup Tests: types, #fields, names, repeated fields
  DAWN_EXPECT_NE(parse("float b = 1; a = b", field("a"), var("b")),
                 block(vardecl("int", "b", lit("1")), assign(field("a"), var("b"))));
  DAWN_EXPECT_NE(parse("a = b", field("a"), field("b"), field("c")),
                 assign(field("a"), field("b")));
  DAWN_EXPECT_NE(parse("a = c", field("a"), field("c")), assign(field("a"), field("b")));
  DAWN_EXPECT_EQ(parse("a = b ; b = a", field("a"), field("b")),
                 block(assign(field("a"), field("b")), assign(field("b"), field("a"))));
}

TEST(ParsingTest, Assignment) {
  // Field - Field
  DAWN_EXPECT_EQ(parse("a = b", field("a"), field("b")), assign(field("a"), field("b")));
  DAWN_EXPECT_EQ(parse("a += b", field("a"), field("b")), assign(field("a"), field("b"), "+="));
  DAWN_EXPECT_EQ(parse("a -= b", field("a"), field("b")), assign(field("a"), field("b"), "-="));
  DAWN_EXPECT_EQ(parse("a *= b", field("a"), field("b")), assign(field("a"), field("b"), "*="));
  DAWN_EXPECT_EQ(parse("a /= b", field("a"), field("b")), assign(field("a"), field("b"), "/="));
  DAWN_EXPECT_NE(parse("a = b", field("a"), field("b")), assign(field("a"), field("c"), "+="));

  //  Field - Variable
  DAWN_EXPECT_EQ(parse("float b = 1; a = b", field("a"), var("b")),
                 block(vardecl("float", "b", lit("1")), assign(field("a"), var("b"))));
  DAWN_EXPECT_EQ(parse("float b = 1; a += b", field("a"), var("b")),
                 block(vardecl("float", "b", lit("1")), assign(field("a"), var("b"), "+=")));
  DAWN_EXPECT_EQ(parse("float b = 1; a -= b", field("a"), var("b")),
                 block(vardecl("float", "b", lit("1")), assign(field("a"), var("b"), "-=")));
  DAWN_EXPECT_EQ(parse("float b = 1; a *= b", field("a"), var("b")),
                 block(vardecl("float", "b", lit("1")), assign(field("a"), var("b"), "*=")));
  DAWN_EXPECT_EQ(parse("float b = 1; a /= b", field("a"), var("b")),
                 block(vardecl("float", "b", lit("1")), assign(field("a"), var("b"), "/=")));

  // Variable - Field
  DAWN_EXPECT_EQ(parse("float b = 1; b = a", field("a"), var("b")),
                 block(vardecl("float", "b", lit("1")), assign(var("b"), field("a"))));
  DAWN_EXPECT_EQ(parse("float b = 1; b += a", field("a"), var("b")),
                 block(vardecl("float", "b", lit("1")), assign(var("b"), field("a"), "+=")));
  DAWN_EXPECT_EQ(parse("float b = 1; b -= a", field("a"), var("b")),
                 block(vardecl("float", "b", lit("1")), assign(var("b"), field("a"), "-=")));
  DAWN_EXPECT_EQ(parse("float b = 1; b *= a", field("a"), var("b")),
                 block(vardecl("float", "b", lit("1")), assign(var("b"), field("a"), "*=")));
  DAWN_EXPECT_EQ(parse("float b = 1; b /= a", field("a"), var("b")),
                 block(vardecl("float", "b", lit("1")), assign(var("b"), field("a"), "/=")));

  // Variable - Variable
  DAWN_EXPECT_EQ(
      parse("float b = 1; float c = 2; b = c; field = b", field("field"), var("b"), var("c")),
      block(vardecl("float", "b", lit("1")), vardecl("float", "c", lit("2")),
            assign(var("b"), var("c")), assign(field("field"), var("b"))));
  DAWN_EXPECT_EQ(
      parse("float b = 1; float c = 2; b += c; field = b", field("field"), var("b"), var("c")),
      block(vardecl("float", "b", lit("1")), vardecl("float", "c", lit("2")),
            assign(var("b"), var("c"), "+="), assign(field("field"), var("b"))));
  DAWN_EXPECT_EQ(
      parse("float b = 1; float c = 2; b -= c; field = b", field("field"), var("b"), var("c")),
      block(vardecl("float", "b", lit("1")), vardecl("float", "c", lit("2")),
            assign(var("b"), var("c"), "-="), assign(field("field"), var("b"))));
  DAWN_EXPECT_EQ(
      parse("float b = 1; float c = 2; b *= c; field = b", field("field"), var("b"), var("c")),
      block(vardecl("float", "b", lit("1")), vardecl("float", "c", lit("2")),
            assign(var("b"), var("c"), "*="), assign(field("field"), var("b"))));
  DAWN_EXPECT_EQ(
      parse("float b = 1; float c = 2; b /= c; field = b", field("field"), var("b"), var("c")),
      block(vardecl("float", "b", lit("1")), vardecl("float", "c", lit("2")),
            assign(var("b"), var("c"), "/="), assign(field("field"), var("b"))));
}

TEST(ParsingTest, Unop) {
  DAWN_EXPECT_EQ(parse("float a = 1; st = ++a;", field("st")),
                 block(vardecl("float", "a", lit("1")), assign(field("st"), unop(var("a"), "++"))));
  DAWN_EXPECT_EQ(parse("float a = 1; st = --a;", field("st")),
                 block(vardecl("float", "a", lit("1")), assign(field("st"), unop(var("a"), "--"))));
  DAWN_EXPECT_EQ(parse("float a = 1; st = -a;", field("st")),
                 block(vardecl("float", "a", lit("1")), assign(field("st"), unop(var("a"), "-"))));
  DAWN_EXPECT_EQ(parse("bool a = 1; st = !a;", field("st")),
                 block(vardecl("bool", "a", lit("1")), assign(field("st"), unop(var("a"), "!"))));
  DAWN_EXPECT_NE(parse("bool a = 1; st = !a;", field("st")),
                 block(vardecl("bool", "a", lit("1")), assign(field("st"), unop(var("a"), "-"))));
}

TEST(ParsingTest, BinOp) {
  // Field - Field operators
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

  // Field - Var operators
  DAWN_EXPECT_EQ(
      parse("float c = 1; a = b + c", field("a"), field("b"), var("c")),
      block(vardecl("float", "c", lit("1")), assign(field("a"), binop(field("b"), "+", var("c")))));
  DAWN_EXPECT_EQ(
      parse("float c = 1; a = b - c", field("a"), field("b"), var("c")),
      block(vardecl("float", "c", lit("1")), assign(field("a"), binop(field("b"), "-", var("c")))));
  DAWN_EXPECT_EQ(
      parse("float c = 1; a = b * c", field("a"), field("b"), var("c")),
      block(vardecl("float", "c", lit("1")), assign(field("a"), binop(field("b"), "*", var("c")))));
  DAWN_EXPECT_EQ(
      parse("float c = 1; a = b / c", field("a"), field("b"), var("c")),
      block(vardecl("float", "c", lit("1")), assign(field("a"), binop(field("b"), "/", var("c")))));

  DAWN_EXPECT_EQ(parse(R"(
                       int b = 1;
                       int c = 2;
                       a = b & c;)",
                       field("a"), var("b"), var("c")),
                 block(vardecl("int", "b", lit("1")), vardecl("int", "c", lit("2")),
                       assign(field("a"), binop(var("b"), "&", var("c")))));
  DAWN_EXPECT_EQ(parse(R"(
                       int b = 1;
                       int c = 2;
                       a = b | c;)",
                       field("a"), var("b"), var("c")),
                 block(vardecl("int", "b", lit("1")), vardecl("int", "c", lit("2")),
                       assign(field("a"), binop(var("b"), "|", var("c")))));
  DAWN_EXPECT_EQ(parse(R"(
                       int b = 1;
                       int c = 2;
                       a = b ^ c;)",
                       field("a"), var("b"), var("c")),
                 block(vardecl("int", "b", lit("1")), vardecl("int", "c", lit("2")),
                       assign(field("a"), binop(var("b"), "^", var("c")))));
  DAWN_EXPECT_EQ(parse(R"(
                        int b = 1;
                        int c = 2;
                        a = b << c;)",
                       field("a"), var("b"), var("c")),
                 block(vardecl("int", "b", lit("1")), vardecl("int", "c", lit("2")),
                       assign(field("a"), binop(var("b"), "<<", var("c")))));
  DAWN_EXPECT_EQ(parse(R"(
                       int b = 1;
                       int c = 2;
                       a = b >> c
                       )",
                       field("a"), var("b"), var("c")),
                 block(vardecl("int", "b", lit("1")), vardecl("int", "c", lit("2")),
                       assign(field("a"), binop(var("b"), ">>", var("c")))));
}

TEST(ParsingTest, TernOp) {
  DAWN_EXPECT_EQ(
      parse(R"(
                       float a = 0;
                       float b = 1;
                       float c = 2;
                       f = a<0 ? b : c;)",
            field("f"), var("a"), var("b"), var("c")),
      block(vardecl("float", "a", lit("0")), vardecl("float", "b", lit("1")),
            vardecl("float", "c", lit("2")),
            assign(field("f"), ternop(binop(var("a"), "<", lit("0")), var("b"), var("c")))));
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
                 block(vardecl("float", "a", lit("0")), vardecl("float", "b", lit("1")),
                       ifstmt(expr(binop(var("b"), ">", lit("0"))),
                              block(expr(assign(field("fieldOne"), var("a")))),
                              block(expr(assign(field("fieldTwo"), var("b")))))));
  DAWN_EXPECT_EQ(parse(R"(
                      float a = 0;
                      float b = 1;
                      if(b > 0){
                      field = a;
                      })",
                       field("field")),
                 block(vardecl("float", "a", lit("0")), vardecl("float", "b", lit("1")),
                       ifstmt(expr(binop(var("b"), ">", lit("0"))),
                              block(expr(assign(field("field"), var("a")))))));
  // Only one else Stmt
  DAWN_EXPECT_NE(parse(R"(
                       float a = 0;
                       float b = 1;
                       if(b > 0){
                       field = a;
                       })",
                       field("field")),
                 block(vardecl("float", "a", lit("0")), vardecl("float", "b", lit("1")),
                       ifstmt(expr(binop(var("b"), ">", lit("0"))),
                              block(expr(assign(field("field"), var("a"))))),
                       block(expr(assign(field("b"), var("field"))))));
  // Else does not match
  DAWN_EXPECT_NE(parse(R"(
                           float a = 0;
                           float b = 1;
                           if(b > 0){
                           fieldOne = a;
                           } else {
                           fieldTwo = b;
                           })",
                       field("fieldOne"), field("fieldTwo")),
                 block(vardecl("float", "a", lit("0")), vardecl("float", "b", lit("1")),
                       ifstmt(expr(binop(var("b"), ">", lit("0"))),
                              block(expr(assign(field("fieldOne"), var("a")))),
                              block(expr(assign(var("b"), field("fieldTwo")))))));
  // If does not match
  DAWN_EXPECT_NE(parse(R"(
                        float a = 0;
                        float b = 1;
                        if(b > 0){
                        fieldOne = a;
                        } else {
                        fieldTwo = b;
                        })",
                       field("fieldOne"), field("fieldTwo")),
                 block(vardecl("float", "a", lit("0")), vardecl("float", "b", lit("1")),
                       ifstmt(expr(binop(var("b"), ">", lit("0"))),
                              block(expr(assign(field("fieldOne"), var("b")))),
                              block(expr(assign(field("fieldTwo"), var("b")))))));
  // condition does not match
  DAWN_EXPECT_NE(parse(R"(
                         float a = 0;
                         float b = 1;
                         if(b > 0){
                         fieldOne = a;
                         } else {
                         fieldTwo = b;
                         })",
                       field("fieldOne"), field("fieldTwo")),
                 block(vardecl("float", "a", lit("0")), vardecl("float", "b", lit("1")),
                       ifstmt(expr(binop(var("a"), ">", lit("0"))),
                              block(expr(assign(field("fieldOne"), var("a")))),
                              block(expr(assign(field("fieldTwo"), var("b")))))));
}

#undef DAWN_EXPECT_EQ

#undef DAWN_EXPECT_NE

} // anonymous namespace
