//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gsl/SIR/ASTVisitor.h"
#include "gsl/SIR/SIR.h"
#include "gsl/SIR/SIRSerializerJSON.h"
#include "gsl/Support/Casting.h"
#include "gsl/Support/Json.h"
#include "gsl/Support/Logging.h"
#include "gsl/Support/Unreachable.h"
#include <fstream>
#include <iostream>
#include <stack>

namespace gsl {

namespace {

json::json TypeToJSON(const Type& type) {
  json::json jType;
  jType["name"] = type.getName();
  jType["builtinTypeID"] = int(type.getBuiltinTypeID());
  jType["cvQualifier"] = int(type.getCVQualifier());
  return jType;
}

json::json SourceLocationToJSON(const SourceLocation& loc) {
  json::json jSourceLocation;
  jSourceLocation["line"] = loc.Line;
  jSourceLocation["column"] = loc.Column;
  return jSourceLocation;
}

json::json IntervalToJSON(const sir::Interval& interval) {
  json::json jInterval;
  jInterval["lowerLevel"] = interval.LowerLevel;
  jInterval["upperLevel"] = interval.UpperLevel;
  jInterval["lowerOffset"] = interval.LowerOffset;
  jInterval["upperOffset"] = interval.UpperOffset;
  return jInterval;
}

json::json FieldToJSON(const sir::Field& field) {
  json::json jField;
  jField["name"] = field.Name;
  jField["loc"] = SourceLocationToJSON(field.Loc);
  jField["isTemporary"] = field.IsTemporary;
  return jField;
}

json::json DirectionToJSON(const sir::Direction& dir) {
  json::json jDirection;
  jDirection["name"] = dir.Name;
  jDirection["loc"] = SourceLocationToJSON(dir.Loc);
  return jDirection;
}

json::json OffsetToJSON(const sir::Offset& off) {
  json::json jOffset;
  jOffset["name"] = off.Name;
  jOffset["loc"] = SourceLocationToJSON(off.Loc);
  return jOffset;
}

class JSONASTSerializer : public ASTVisitor {
  json::json jAST_;
  std::stack<std::reference_wrapper<json::json>> curNodes_;

public:
  JSONASTSerializer() : jAST_(json::json()) { curNodes_.push(jAST_); }

  /// @brief Get the JSON representation of the AST
  const json::json& getAST() const { return jAST_; }

  /// @brief Get the current node
  json::json& getCurrentNode() { return curNodes_.top().get(); }

  /// @brief Parse an expression or statement
  template <class T>
  json::json parse(const std::shared_ptr<T>& node) {
    json::json jNode;
    curNodes_.push(jNode);
    node->accept(*this);
    curNodes_.pop();
    return jNode;
  }

  void visit(const std::shared_ptr<BlockStmt>& stmt) override {
    json::json jBlockStmt;

    for(const auto& s : stmt->getStatements())
      jBlockStmt["statements"].push_back(parse(s));

    getCurrentNode()["BlockStmt"] = jBlockStmt;
  }

  void visit(const std::shared_ptr<ExprStmt>& stmt) override {
    json::json jExprStmt;
    jExprStmt["loc"] = SourceLocationToJSON(stmt->getSourceLocation());
    jExprStmt["expr"] = parse(stmt->getExpr());

    getCurrentNode()["ExprStmt"] = jExprStmt;
  }

  void visit(const std::shared_ptr<ReturnStmt>& stmt) override {
    json::json jReturnStmt;
    jReturnStmt["loc"] = SourceLocationToJSON(stmt->getSourceLocation());
    jReturnStmt["expr"] = parse(stmt->getExpr());
    ;

    getCurrentNode()["ReturnStmt"] = jReturnStmt;
  }

  void visit(const std::shared_ptr<VarDeclStmt>& stmt) override {
    json::json jVarDeclStmt;

    jVarDeclStmt["loc"] = SourceLocationToJSON(stmt->getSourceLocation());
    jVarDeclStmt["name"] = stmt->getName();
    jVarDeclStmt["type"] = TypeToJSON(stmt->getType());
    jVarDeclStmt["dimension"] = stmt->getDimension();
    jVarDeclStmt["op"] = stmt->getOp();

    if(stmt->hasInit())
      jVarDeclStmt["initList"] = json::json();
    else
      for(const auto& e : stmt->getInitList())
        jVarDeclStmt["initList"].push_back(parse(e));

    getCurrentNode()["VarDeclStmt"] = jVarDeclStmt;
  }

  virtual void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) override {
    json::json jVerticalRegionDeclStmt;

    jVerticalRegionDeclStmt["loc"] = SourceLocationToJSON(stmt->getSourceLocation());
    jVerticalRegionDeclStmt["interval"] =
        IntervalToJSON(*stmt->getVerticalRegion()->VerticalInterval);
    jVerticalRegionDeclStmt["loopOrder"] =
        stmt->getVerticalRegion()->LoopOrder == sir::VerticalRegion::LK_Forward ? "forward"
                                                                                : "backward";

    JSONASTSerializer serializer;
    stmt->getVerticalRegion()->Ast->accept(serializer);
    jVerticalRegionDeclStmt["ast"] = serializer.getAST();

    getCurrentNode()["VerticalRegionDeclStmt"] = jVerticalRegionDeclStmt;
  }

  virtual void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override {
    json::json jStencilCallDeclStmt;

    jStencilCallDeclStmt["loc"] = SourceLocationToJSON(stmt->getSourceLocation());
    jStencilCallDeclStmt["callee"] = stmt->getStencilCall()->Callee;

    for(const auto& arg : stmt->getStencilCall()->Args)
      jStencilCallDeclStmt["args"].push_back(FieldToJSON(*arg));

    getCurrentNode()["StencilCallDeclStmt"] = jStencilCallDeclStmt;
  }

  virtual void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) override {
    GSL_ASSERT_MSG(0, "not yet implemented");
  }

  void visit(const std::shared_ptr<IfStmt>& stmt) override {
    json::json jIfStmt;

    jIfStmt["cond"] = parse(stmt->getCondStmt());
    jIfStmt["then"] = parse(stmt->getThenStmt());
    jIfStmt["else"] = stmt->hasElse() ? parse(stmt->getElseStmt()) : json::json();
    getCurrentNode()["IfStmt"] = jIfStmt;
  }

  void visit(const std::shared_ptr<UnaryOperator>& expr) override {
    json::json jUnaryOperator;

    jUnaryOperator["loc"] = SourceLocationToJSON(expr->getSourceLocation());
    jUnaryOperator["operand"] = parse(expr->getOperand());
    jUnaryOperator["op"] = expr->getOp();
    getCurrentNode()["UnaryOperator"] = jUnaryOperator;
  }

  void visit(const std::shared_ptr<BinaryOperator>& expr) override {
    json::json jBinaryOperator;

    jBinaryOperator["loc"] = SourceLocationToJSON(expr->getSourceLocation());
    jBinaryOperator["left"] = parse(expr->getLeft());
    jBinaryOperator["right"] = parse(expr->getRight());
    jBinaryOperator["op"] = expr->getOp();
    getCurrentNode()["BinaryOperator"] = jBinaryOperator;
  }

  void visit(const std::shared_ptr<AssignmentExpr>& expr) override {
    json::json jAssignmentExpr;

    jAssignmentExpr["loc"] = SourceLocationToJSON(expr->getSourceLocation());
    jAssignmentExpr["left"] = parse(expr->getLeft());
    jAssignmentExpr["right"] = parse(expr->getRight());
    jAssignmentExpr["op"] = expr->getOp();
    getCurrentNode()["AssignmentExpr"] = jAssignmentExpr;
  }

  void visit(const std::shared_ptr<TernaryOperator>& expr) override {
    json::json jTernaryOperator;

    jTernaryOperator["loc"] = SourceLocationToJSON(expr->getSourceLocation());
    jTernaryOperator["cond"] = parse(expr->getCondition());
    jTernaryOperator["left"] = parse(expr->getLeft());
    jTernaryOperator["right"] = parse(expr->getRight());
    getCurrentNode()["TernaryOperator"] = jTernaryOperator;
  }

  void visit(const std::shared_ptr<FunCallExpr>& expr) override {
    json::json jFunCallExpr;

    jFunCallExpr["loc"] = SourceLocationToJSON(expr->getSourceLocation());
    jFunCallExpr["callee"] = expr->getCallee();

    for(std::size_t i = 0; i < expr->getArguments().size(); ++i)
      jFunCallExpr["arguments"].push_back(parse(expr->getArguments()[i]));

    getCurrentNode()["FunCallExpr"] = jFunCallExpr;
  }

  void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override {
    json::json jStencilFunCallExpr;

    jStencilFunCallExpr["loc"] = SourceLocationToJSON(expr->getSourceLocation());
    jStencilFunCallExpr["callee"] = expr->getCallee();

    for(std::size_t i = 0; i < expr->getArguments().size(); ++i)
      jStencilFunCallExpr["arguments"].push_back(parse(expr->getArguments()[i]));

    getCurrentNode()["StencilFunCallExpr"] = jStencilFunCallExpr;
  }

  void visit(const std::shared_ptr<StencilFunArgExpr>& expr) override {
    json::json jStencilFunArgExpr;

    jStencilFunArgExpr["loc"] = SourceLocationToJSON(expr->getSourceLocation());
    jStencilFunArgExpr["dimension"] = expr->getDimension();
    jStencilFunArgExpr["offset"] = expr->getOffset();
    jStencilFunArgExpr["argumentIndex"] = expr->getArgumentIndex();
    getCurrentNode()["StencilFunArgExpr"] = jStencilFunArgExpr;
  }

  void visit(const std::shared_ptr<VarAccessExpr>& expr) override {
    json::json jVarAccessExpr;

    jVarAccessExpr["loc"] = SourceLocationToJSON(expr->getSourceLocation());
    jVarAccessExpr["name"] = expr->getName();
    jVarAccessExpr["index"] = expr->isArrayAccess() ? parse(expr->getIndex()) : json::json();
    jVarAccessExpr["isExternal"] = expr->isExternal();
    getCurrentNode()["VarAccessExpr"] = jVarAccessExpr;
  }

  void visit(const std::shared_ptr<FieldAccessExpr>& expr) override {
    json::json jFieldAccessExpr;

    jFieldAccessExpr["loc"] = SourceLocationToJSON(expr->getSourceLocation());
    jFieldAccessExpr["name"] = expr->getName();
    jFieldAccessExpr["offset"] = expr->getOffset();
    jFieldAccessExpr["argumentMap"] = expr->getArgumentMap();
    jFieldAccessExpr["argumentOffset"] = expr->getArgumentOffset();
    jFieldAccessExpr["negateOffset"] = expr->negateOffset();

    getCurrentNode()["FieldAccessExpr"] = jFieldAccessExpr;
  }

  void visit(const std::shared_ptr<LiteralAccessExpr>& expr) override {
    json::json jLiteralAccessExpr;

    jLiteralAccessExpr["loc"] = SourceLocationToJSON(expr->getSourceLocation());
    jLiteralAccessExpr["value"] = expr->getValue();
    jLiteralAccessExpr["builtinType"] = int(expr->getBuiltinType());
    getCurrentNode()["LiteralAccessExpr"] = jLiteralAccessExpr;
  }
};

} // anonymous namespace

std::shared_ptr<SIR> SIRSerializerJSON::deserialize(const std::string& file) { return nullptr; }

void SIRSerializerJSON::serialize(const std::string& file, const SIR* sir) {
  json::json jFile;

  jFile["filename"] = sir->Filename;

  // Serialize the stencil-functions
  for(const auto& stencilFunction : sir->StencilFunctions) {
    json::json jStencilFunction;

    jStencilFunction["name"] = stencilFunction->Name;
    for(const auto& arg : stencilFunction->Args) {
      json::json jArg;

      if(sir::Field* field = dyn_cast<sir::Field>(arg.get())) {
        jArg["value"] = FieldToJSON(*field);
        jArg["type"] = "field";
      } else if(sir::Direction* dir = dyn_cast<sir::Direction>(arg.get())) {
        jArg["value"] = DirectionToJSON(*dir);
        jArg["type"] = "direction";
      } else if(sir::Offset* offset = dyn_cast<sir::Offset>(arg.get())) {
        jArg["value"] = OffsetToJSON(*offset);
        jArg["type"] = "offset";
      }

      jStencilFunction["arguments"].push_back(jArg);
    }

    
    if(stencilFunction->Intervals.empty())
      jStencilFunction["intervals"] = json::json();      
    else
      for(const auto& interval : stencilFunction->Intervals)
        jStencilFunction["intervals"].push_back(IntervalToJSON(*interval));
    
    for(const auto& ast : stencilFunction->Asts) {
      JSONASTSerializer serializer;
      ast->accept(serializer);
      jStencilFunction["asts"].push_back(serializer.getAST());
    }

    jFile["stencilFunctions"].push_back(jStencilFunction);
  }

  // Serialize the stencils
  for(const auto& stencil : sir->Stencils) {
    json::json jStencil;

    jStencil["name"] = stencil->Name;

    for(const auto& field : stencil->Fields)
      jStencil["fields"].push_back(FieldToJSON(*field));

    JSONASTSerializer serializer;
    stencil->StencilDescAst->accept(serializer);
    jStencil["ast"] = serializer.getAST();

    jFile["stencils"].push_back(jStencil);
  }

  // Serialize the globals map
  for(const auto& nameValuePair : *(sir->GlobalVariableMap)) {
    json::json variable;

    sir::Value& value = *nameValuePair.second;
    variable["type"] = sir::Value::typeToString(value.getType());

    if(value.empty()) {
      variable["value"] = json::json();
    } else {
      switch(value.getType()) {
      case sir::Value::Boolean:
        variable["value"] = value.getValue<bool>();
        break;
      case sir::Value::Integer:
        variable["value"] = value.getValue<int>();
        break;
      case sir::Value::Double:
        variable["value"] = value.getValue<double>();
        break;
      case sir::Value::String:
        variable["value"] = value.getValue<std::string>();
        break;
      default:
        gsl_unreachable("invalid type");
      }
    }

    jFile["globals"][nameValuePair.first] = variable;
  }

  std::ofstream ofs(file);
  if(!ofs.is_open()) {
    GSL_LOG(ERROR) << "Failed to open JSON file \"" << file << "\" for writing";
  } else {
    ofs << jFile.dump(2) << std::endl;
    ofs.close();
  }
}

} // namespace gsl
