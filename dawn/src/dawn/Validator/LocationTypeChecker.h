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

#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTFwd.h"
#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include <memory>

namespace dawn {
class LocationTypeChecker {

private:
  class TypeCheckerImpl : public ast::ASTVisitorForwarding {
    std::optional<ast::Expr::LocationType> curType_;
    const std::unordered_map<std::string, ast::Expr::LocationType> nameToLocationType_;
    const std::unordered_map<int, std::string> idToNameMap_;
    bool typesConsistent_ = true;

  public:
    void visit(const std::shared_ptr<iir::FieldAccessExpr>& stmt) override;
    void visit(const std::shared_ptr<iir::BinaryOperator>& stmt) override;
    void visit(const std::shared_ptr<iir::AssignmentExpr>& stmt) override;
    void visit(const std::shared_ptr<iir::ReductionOverNeighborExpr>& stmt) override;

    bool isConsistent() const { return typesConsistent_; }
    bool hasType() const { return curType_.has_value(); };
    ast::Expr::LocationType getType() const;

    // This constructor is used when the check is performed on the SIR. In this case, each
    // Field is uniquely identified by its name
    TypeCheckerImpl(
        const std::unordered_map<std::string, ast::Expr::LocationType> nameToLocationMap);
    // This constructor is used when the check is performed from IIR. In this case, the fields may
    // have been renamed if stencils had to be merged. Hence, an additional map with key AccessID is
    // needed
    TypeCheckerImpl(
        const std::unordered_map<std::string, ast::Expr::LocationType> nameToLocationMap,
        const std::unordered_map<int, std::string> idToNameMap);
  };

public:
  bool checkLocationTypeConsistency(const dawn::SIR&);
  bool checkLocationTypeConsistency(const dawn::iir::IIR&, const iir::StencilMetaInformation&);
};
} // namespace dawn