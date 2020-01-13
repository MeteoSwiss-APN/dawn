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

#pragma once

#include "dawn/IIR/ASTFwd.h"
#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/IIRNodeIterator.h"

namespace dawn {
class TypeChecker {

private:
  class TypeCheckerImpl : public ast::ASTVisitorForwarding {
    std::optional<ast::Expr::LocationType> curType_;
    const std::unordered_map<std::string, ast::Expr::LocationType>& nameToLocationType_;
    bool typesConsistent_ = true;

  public:
    void visit(const std::shared_ptr<iir::FieldAccessExpr>& stmt) override;
    void visit(const std::shared_ptr<iir::BinaryOperator>& stmt) override;
    void visit(const std::shared_ptr<iir::AssignmentExpr>& stmt) override;
    void visit(const std::shared_ptr<iir::ReductionOverNeighborExpr>& stmt) override;

    bool isConsistent() const { return typesConsistent_; }
    bool hasType() const { return curType_.has_value(); };
    ast::Expr::LocationType getType() const;

    TypeCheckerImpl(
        const std::unordered_map<std::string, ast::Expr::LocationType>& nameToLocationMap);
  };

public:
  bool checkLocationTypeConsistency(const dawn::SIR&);
  bool checkLocationTypeConsistency(const dawn::iir::IIR&);
};
} // namespace dawn