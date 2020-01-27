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
class UnstructuredDimensionChecker {

private:
  class UnstructuredDimensionCheckerImpl : public ast::ASTVisitorForwarding {
    std::optional<sir::FieldDimensions> curDimensions_;
    const std::unordered_map<std::string, sir::FieldDimensions> nameToDimensions_;
    const std::unordered_map<int, std::string> idToNameMap_;
    bool dimensionsConsistent_ = true;

    void checkBinaryOpUnstructured(const sir::FieldDimensions& left,
                                   const sir::FieldDimensions& right);

  public:
    void visit(const std::shared_ptr<iir::FieldAccessExpr>& stmt) override;
    void visit(const std::shared_ptr<iir::BinaryOperator>& stmt) override;
    void visit(const std::shared_ptr<iir::AssignmentExpr>& stmt) override;
    void visit(const std::shared_ptr<iir::ReductionOverNeighborExpr>& stmt) override;

    bool isConsistent() const { return dimensionsConsistent_; }
    bool hasDimensions() const { return curDimensions_.has_value(); };
    const sir::FieldDimensions& getDimensions() const;

    // This constructor is used when the check is performed on the SIR. In this case, each
    // Field is uniquely identified by its name
    UnstructuredDimensionCheckerImpl(
        const std::unordered_map<std::string, sir::FieldDimensions> nameToDimensionsMap);
    // This constructor is used when the check is performed from IIR. In this case, the fields may
    // have been renamed if stencils had to be merged. Hence, an additional map with key AccessID is
    // needed
    UnstructuredDimensionCheckerImpl(
        const std::unordered_map<std::string, sir::FieldDimensions> nameToDimensionsMap,
        const std::unordered_map<int, std::string> idToNameMap);
  };

public:
  bool checkDimensionsConsistency(const dawn::SIR&);
  bool checkDimensionsConsistency(const dawn::iir::IIR&, const iir::StencilMetaInformation&);
};
} // namespace dawn