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

#include "dawn/AST/ASTVisitor.h"
#include "dawn/IIR/IIR.h"
#include "dawn/Support/SourceLocation.h"

#include <memory>

namespace dawn {

class IndirectionChecker {

  class IndirectionCheckerImpl : public ast::ASTVisitorForwarding {
  private:
    bool indirectionsValid_ = true;
    bool lhs_ = false;

  public:
    void visit(const std::shared_ptr<ast::FieldAccessExpr>& expr);
    void visit(const std::shared_ptr<ast::AssignmentExpr>& expr);
    bool indirectionsAreValid() const { return indirectionsValid_; }
  };

public:
  using IndirectionResult = std::tuple<bool, SourceLocation>;
  static IndirectionResult checkIndirections(const dawn::SIR&);
  static IndirectionResult checkIndirections(const dawn::iir::IIR&);
};

} // namespace dawn