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
#include "dawn/IIR/AST.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/Support/Unreachable.h"
#include <string>

namespace dawn {
namespace codegen {
namespace cuda {

ASTStencilBody::ASTStencilBody(const iir::StencilMetaInformation& metadata,
                               const std::unordered_map<int, Array3i>& fieldIndexMap,
                               const std::unique_ptr<iir::MultiStage>& ms,
                               const CacheProperties& cacheProperties, Array3ui blockSizes)
    : ASTCodeGenCXX(), metadata_(metadata), offsetPrinter_("+", "", "", true),
      fieldIndexMap_(fieldIndexMap), ms_(ms), cacheProperties_(cacheProperties),
      blockSizes_(blockSizes) {}

ASTStencilBody::~ASTStencilBody() {}

std::string ASTStencilBody::getName(const std::shared_ptr<iir::Expr>& expr) const {
  return metadata_.getFieldNameFromAccessID(iir::getAccessID(expr));
}

std::string ASTStencilBody::getName(const std::shared_ptr<iir::VarDeclStmt>& stmt) const {
  return metadata_.getFieldNameFromAccessID(iir::getAccessID(stmt));
}

//===------------------------------------------------------------------------------------------===//
//     Stmt
//===------------------------------------------------------------------------------------------===//

void ASTStencilBody::visit(const std::shared_ptr<iir::ReturnStmt>& stmt) {
  if(scopeDepth_ == 0)
    ss_ << std::string(indent_, ' ');

  ss_ << "return ";

  stmt->getExpr()->accept(*this);
  ss_ << ";\n";
}

void ASTStencilBody::visit(const std::shared_ptr<iir::VerticalRegionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "VerticalRegionDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<iir::StencilCallDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "StencilCallDeclStmt not allowed in this context");
}

void ASTStencilBody::visit(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt) {
  DAWN_ASSERT_MSG(0, "BoundaryConditionDeclStmt not allowed in this context");
}

//===------------------------------------------------------------------------------------------===//
//     Expr
//===------------------------------------------------------------------------------------------===//

void ASTStencilBody::visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) {
  dawn_unreachable("stencil functions not allows in cuda backend");
}

void ASTStencilBody::visit(const std::shared_ptr<iir::StencilFunArgExpr>& expr) {
  dawn_unreachable("stencil functions not allows in cuda backend");
}

void ASTStencilBody::visit(const std::shared_ptr<iir::VarAccessExpr>& expr) {
  std::string name = getName(expr);
  int accessID = iir::getAccessID(expr);

  if(metadata_.isAccessType(iir::FieldAccessType::GlobalVariable, accessID)) {
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

void ASTStencilBody::visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) {
  int accessID = iir::getAccessID(expr);
  if(cacheProperties_.isIJCached(accessID)) {
    derefIJCache(expr);
    return;
  }
  if(cacheProperties_.isKCached(accessID)) {
    derefKCache(expr);
    return;
  }

  CodeGeneratorHelper::generateFieldAccessDeref(ss_, ms_, metadata_, accessID, fieldIndexMap_,
                                                expr->getOffset());
}

void ASTStencilBody::derefIJCache(const std::shared_ptr<iir::FieldAccessExpr>& expr) {
  int accessID = iir::getAccessID(expr);
  std::string accessName = cacheProperties_.getCacheName(accessID);

  std::string index;
  if(cacheProperties_.isCommonCache(accessID)) {
    index = cacheProperties_.getCommonCacheIndexName(iir::Cache::CacheType::IJ);
  } else {
    index = "iblock - " + std::to_string(cacheProperties_.getOffsetBeginIJCache(accessID, 0)) +
            " + (jblock - " + std::to_string(cacheProperties_.getOffsetBeginIJCache(accessID, 1)) +
            ")*" + std::to_string(cacheProperties_.getStride(accessID, 1, blockSizes_));
  }
  auto offset = expr->getOffset();
  DAWN_ASSERT(offset.verticalOffset() == 0);
  auto const& hoffset = ast::offset_cast<ast::CartesianOffset const&>(offset.horizontalOffset());

  std::string offsetStr;
  if(hoffset.offsetI() != 0)
    offsetStr += std::to_string(hoffset.offsetI());
  if(hoffset.offsetJ() != 0)
    offsetStr += ((offsetStr != "") ? "+" : "") + std::to_string(hoffset.offsetJ()) + "*" +
                 std::to_string(cacheProperties_.getStride(accessID, 1, blockSizes_));
  ss_ << accessName
      << (offsetStr.empty() ? "[" + index + "]" : ("[" + index + "+" + offsetStr + "]"));
}

void ASTStencilBody::derefKCache(const std::shared_ptr<iir::FieldAccessExpr>& expr) {
  int accessID = iir::getAccessID(expr);
  std::string accessName = cacheProperties_.getCacheName(accessID);
  [[maybe_unused]] auto vertExtent = ms_->getKCacheVertExtent(accessID);

  const int kcacheCenterOffset = cacheProperties_.getKCacheCenterOffset(accessID);

  auto offset = expr->getOffset();
  [[maybe_unused]] auto const& hoffset =
      ast::offset_cast<ast::CartesianOffset const&>(offset.horizontalOffset());

  DAWN_ASSERT(hoffset.offsetI() == 0 && hoffset.offsetJ() == 0);
  DAWN_ASSERT((offset.verticalOffset() <= vertExtent.plus()) &&
              (offset.verticalOffset() >= vertExtent.minus()));

  int index = kcacheCenterOffset;

  if(offset.verticalOffset() != 0)
    index += offset.verticalOffset();
  ss_ << accessName << "[" + std::to_string(index) + "]";
}

} // namespace cuda
} // namespace codegen
} // namespace dawn
