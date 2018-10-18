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

#include "dawn/CodeGen/Cuda/ASTStencilBody.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/CodeGen/Cuda/ASTStencilFunctionParamVisitor.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/SIR/AST.h"
#include "dawn/Support/Unreachable.h"
#include <string>

namespace dawn {
namespace codegen {
namespace cuda {

ASTStencilBody::ASTStencilBody(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const std::unordered_map<int, Array3i>& fieldIndexMap, const iir::MultiStage& ms,
    const CacheProperties& cacheProperties, Array3ui blockSizes)
    : ASTCodeGenCXX(), instantiation_(stencilInstantiation), offsetPrinter_("+", "", "", true),
      fieldIndexMap_(fieldIndexMap), ms_(ms), cacheProperties_(cacheProperties),
      blockSizes_(blockSizes) {}

ASTStencilBody::~ASTStencilBody() {}

std::string ASTStencilBody::getName(const std::shared_ptr<Expr>& expr) const {
  return instantiation_->getNameFromAccessID(instantiation_->getAccessIDFromExpr(expr));
}

std::string ASTStencilBody::getName(const std::shared_ptr<Stmt>& stmt) const {
  return instantiation_->getNameFromAccessID(instantiation_->getAccessIDFromStmt(stmt));
}

std::array<std::string, 3> ASTStencilBody::ijkfyOffset(const Array3i& offsets, bool isTemporary,
                                                       const Array3i iteratorDims) {
  int n = -1;

  std::array<std::string, 3> res;
  std::transform(offsets.begin(), offsets.end(), res.begin(), [&](int const& off) {
    ++n;
    std::array<std::string, 3> indices{CodeGeneratorHelper::generateStrideName(0, iteratorDims),
                                       CodeGeneratorHelper::generateStrideName(1, iteratorDims),
                                       CodeGeneratorHelper::generateStrideName(2, iteratorDims)};

    if(isTemporary) {
      indices = {"1", "jstride_tmp", "kstride_tmp"};
    }
    if(!(iteratorDims[n]) || !off)
      return std::string("");

    return (indices[n] + "*" + std::to_string(off));
  });
  return res;
}

//===------------------------------------------------------------------------------------------===//
//     Stmt
//===------------------------------------------------------------------------------------------===//

void ASTStencilBody::visit(const std::shared_ptr<ReturnStmt>& stmt) {
  if(scopeDepth_ == 0)
    ss_ << std::string(indent_, ' ');

  ss_ << "return ";

  stmt->getExpr()->accept(*this);
  ss_ << ";\n";
}

void ASTStencilBody::visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "VerticalRegionDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "StencilCallDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "BoundaryConditionDeclStmt not allowed in this context");
}

//===------------------------------------------------------------------------------------------===//
//     Expr
//===------------------------------------------------------------------------------------------===//

void ASTStencilBody::visit(const std::shared_ptr<StencilFunCallExpr>& expr) {
  dawn_unreachable("stencil functions not allows in cuda backend");
}

void ASTStencilBody::visit(const std::shared_ptr<StencilFunArgExpr>& expr) {
  dawn_unreachable("stencil functions not allows in cuda backend");
}

void ASTStencilBody::visit(const std::shared_ptr<VarAccessExpr>& expr) {
  std::string name = getName(expr);
  int accessID = instantiation_->getAccessIDFromExpr(expr);

  if(instantiation_->isGlobalVariable(accessID)) {
    ss_ << "globals_." << name;
  } else {
    ss_ << name;

    if(expr->isArrayAccess()) {
      ss_ << "[";
      expr->getIndex()->accept(*this);
      ss_ << "]";
    }
  }
}

void ASTStencilBody::visit(const std::shared_ptr<FieldAccessExpr>& expr) {
  std::string accessName = getName(expr);
  int accessID = instantiation_->getAccessIDFromExpr(expr);
  if(cacheProperties_.isIJCached(accessID)) {
    derefIJCache(expr);
    return;
  }
  if(cacheProperties_.isKCached(accessID)) {
    derefKCache(expr);
    return;
  }

  bool isTemporary = instantiation_->isTemporaryField(accessID);
  DAWN_ASSERT(fieldIndexMap_.count(accessID) || isTemporary);
  std::string index = isTemporary ? "idx_tmp" : "idx" + CodeGeneratorHelper::indexIteratorName(
                                                            fieldIndexMap_.at(accessID));

  // temporaries have all 3 dimensions
  Array3i iter = isTemporary ? Array3i{1, 1, 1} : fieldIndexMap_.at(accessID);

  std::string offsetStr = offsetPrinter_(ijkfyOffset(expr->getOffset(), isTemporary, iter));
  ss_ << accessName
      << (offsetStr.empty() ? "[" + index + "]" : ("[" + index + "+" + offsetStr + "]"));
}

void ASTStencilBody::derefIJCache(const std::shared_ptr<FieldAccessExpr>& expr) {
  int accessID = instantiation_->getAccessIDFromExpr(expr);
  std::string accessName = cacheProperties_.getCacheName(accessID, instantiation_);

  std::string index;
  if(cacheProperties_.isCommonCache(accessID)) {
    index = cacheProperties_.getCommonCacheIndexName(iir::Cache::CacheTypeKind::IJ);
  } else {
    index = "iblock - " + std::to_string(cacheProperties_.getOffset(accessID, 0)) + " (jblock - " +
            std::to_string(cacheProperties_.getOffset(accessID, 1)) + ")*" +
            std::to_string(cacheProperties_.getStride(accessID, 1, blockSizes_));
  }
  DAWN_ASSERT(expr->getOffset()[2] == 0);

  auto offset = expr->getOffset();
  std::string offsetStr;
  if(offset[0] != 0)
    offsetStr += std::to_string(offset[0]);
  if(offset[1] != 0)
    offsetStr += ((offsetStr != "") ? "+" : "") + std::to_string(offset[1]) + "*" +
                 std::to_string(cacheProperties_.getStride(accessID, 1, blockSizes_));
  ss_ << accessName
      << (offsetStr.empty() ? "[" + index + "]" : ("[" + index + "+" + offsetStr + "]"));
}

void ASTStencilBody::derefKCache(const std::shared_ptr<FieldAccessExpr>& expr) {
  int accessID = instantiation_->getAccessIDFromExpr(expr);
  std::string accessName = cacheProperties_.getCacheName(accessID, instantiation_);
  auto vertExtent = cacheProperties_.getKCacheVertExtent(accessID);

  const int kcacheCenterOffset = cacheProperties_.getKCacheCenterOffset(accessID);

  DAWN_ASSERT((expr->getOffset()[0] == 0) && (expr->getOffset()[1] == 0));
  DAWN_ASSERT((expr->getOffset()[2] <= vertExtent.Plus) &&
              (expr->getOffset()[2] >= vertExtent.Minus));

  int index = kcacheCenterOffset;

  auto offset = expr->getOffset();
  if(offset[2] != 0)
    index += offset[2];
  ss_ << accessName << "[" + std::to_string(index) + "]";
}

} // namespace cuda
} // namespace codegen
} // namespace dawn
