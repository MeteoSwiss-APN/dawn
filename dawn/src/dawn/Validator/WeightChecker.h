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

#include "dawn/AST/ASTExpr.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTFwd.h"
#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include "dawn/Support/SourceLocation.h"
#include <memory>

// This validator checks if all weights in all reduce over neighbor expressions are valid.
//
// Eventually, we want to allow every rvalue as a weight. For now, the interface (to the
// unstructured libraries) is limiting what we can do. This entails that we currently can't generate
// accesses to sparse fields, ReducitonOverNeighborExprs, function calls etc. Basically, only field
// accesses and arithmetic combinations thereof are allowed

namespace dawn {

class WeightChecker {
private:
  class WeightCheckerImpl : public dawn::ast::ASTVisitorForwarding {
  private:
    bool weightsValid_ = true;
    bool parentIsWeight_ = false;
    const std::unordered_map<std::string, sir::FieldDimensions> nameToDimensions_;
    const std::unordered_map<int, std::string> idToNameMap_;

  public:
    void visit(const std::shared_ptr<dawn::ast::FieldAccessExpr>& expr) override;
    void visit(const std::shared_ptr<dawn::ast::FunCallExpr>& expr) override;
    void visit(const std::shared_ptr<dawn::ast::StencilFunCallExpr>& expr) override;
    void visit(const std::shared_ptr<dawn::ast::StencilFunArgExpr>& expr) override;
    void visit(const std::shared_ptr<dawn::ast::ReductionOverNeighborExpr>& expr) override;

    bool isValid() const;

    // This constructor is used when the check is performed on the SIR. In this case, each
    // Field is uniquely identified by its name
    WeightCheckerImpl(
        const std::unordered_map<std::string, sir::FieldDimensions> nameToDimensionsMap);
    // This constructor is used when the check is performed from IIR. In this case, the fields may
    // have been renamed if stencils had to be merged. Hence, an additional map with key AccessID is
    // needed
    WeightCheckerImpl(
        const std::unordered_map<std::string, sir::FieldDimensions> nameToDimensionsMap,
        const std::unordered_map<int, std::string> idToNameMap);
  };

public:
  using ConsistencyResult = std::tuple<bool, SourceLocation>;

  static ConsistencyResult CheckWeights(const iir::IIR& iir,
                                        const iir::StencilMetaInformation& metaInformation);
  static ConsistencyResult CheckWeights(const SIR& sir);
};
} // namespace dawn