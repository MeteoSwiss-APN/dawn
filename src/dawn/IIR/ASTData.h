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

#ifndef DAWN_IIR_ASTDATA_H
#define DAWN_IIR_ASTDATA_H

#include <memory>
#include <vector>

namespace dawn {
namespace ast {
struct StencilCall;
}
namespace iir {

struct IIRASTData {
  //
  // TODO refactor_AST: incrementally add data on top of AST statements to avoid carrying maps, refs
  // to (big) objects, etc. and to keep the information at the lowest possible level of the IIR
  // tree.
  // Need to add: Accesses, (Stmt)Metadata, (Expr)Metadata
  //
  struct StmtData {
    std::shared_ptr<std::vector<ast::StencilCall*>> StackTrace = nullptr;
    StmtData() {}
    StmtData(const StmtData& stmtData) {
      if(stmtData.StackTrace)
        StackTrace = std::make_shared<std::vector<ast::StencilCall*>>(*stmtData.StackTrace);
    }
    StmtData& operator=(StmtData const& stmtData) {
      StackTrace = stmtData.StackTrace;
      return *this;
    }
  };
  struct BlockStmt : virtual public StmtData {};
  struct ExprStmt : virtual public StmtData {};
  struct ReturnStmt : virtual public StmtData {};
  struct VarDeclStmt : virtual public StmtData {};
  struct StencilCallDeclStmt : virtual public StmtData {};
  struct BoundaryConditionDeclStmt : virtual public StmtData {};
  struct IfStmt : virtual public StmtData {};

  struct NOPExpr {};
  struct UnaryOperator {};
  struct BinaryOperator {};
  struct AssignmentExpr {};
  struct TernaryOperator {};
  struct FunCallExpr {};
  struct StencilFunCallExpr {};
  struct StencilFunArgExpr {};
  struct VarAccessExpr {};
  struct FieldAccessExpr {};
  struct LiteralAccessExpr {};
};
} // namespace iir
} // namespace dawn

#endif
