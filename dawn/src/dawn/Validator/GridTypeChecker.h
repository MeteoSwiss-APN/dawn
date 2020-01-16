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

#include "dawn/AST/GridType.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTFwd.h"
#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include <memory>

namespace dawn {
class GridTypeChecker {
private:
  class TypeCheckerImpl : public ast::ASTVisitorForwarding {
    ast::GridType prescribedType_;
    bool typesConsistent_ = true;

  public:
    // currently only the field access expression seems to be important since it uses the Offsets
    // - The Extents are only used in the Accesses which are not part of the AST (but the meta data.
    //   This is still checked, but not using this visitor)
    // - The is currently no unstructured FieldDimensions, hence this is not checked. This will need
    //   to be added as soon as the unstructured version is introduced
    void visit(const std::shared_ptr<iir::FieldAccessExpr>& stmt) override;
    bool isConsistent() const { return typesConsistent_; }

    TypeCheckerImpl(ast::GridType prescribedType) : prescribedType_(prescribedType){};
  };

public:
  bool checkGridTypeConsistency(const dawn::SIR&);
  bool checkGridTypeConsistency(const dawn::iir::IIR&);
};
} // namespace dawn