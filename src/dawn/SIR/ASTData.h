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

#ifndef DAWN_SIR_ASTDATA_H
#define DAWN_SIR_ASTDATA_H

#include <memory>

namespace dawn {
namespace sir {

struct VerticalRegion;

struct SIRASTData {
  struct StmtData {
    StmtData() {}
    StmtData(const StmtData& stmtData) {}
    StmtData& operator=(StmtData const& stmtData) { return *this; }
  };
  struct BlockStmt : virtual public StmtData {};
  struct ExprStmt : virtual public StmtData {};
  struct ReturnStmt : virtual public StmtData {};
  struct VarDeclStmt : virtual public StmtData {};
  struct VerticalRegionDeclStmt : virtual public StmtData {
    const std::shared_ptr<VerticalRegion> verticalRegion_;

    VerticalRegionDeclStmt(const std::shared_ptr<sir::VerticalRegion>& verticalRegion);
    VerticalRegionDeclStmt(const VerticalRegionDeclStmt& stmtData);
    VerticalRegionDeclStmt& operator=(VerticalRegionDeclStmt const& stmtData);
  };
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
} // namespace sir
} // namespace dawn

#endif
