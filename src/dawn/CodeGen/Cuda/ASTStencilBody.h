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

#ifndef DAWN_CODEGEN_CUDA_ASTSTENCILBODY_H
#define DAWN_CODEGEN_CUDA_ASTSTENCILBODY_H

#include "dawn/CodeGen/ASTCodeGenCXX.h"
#include "dawn/CodeGen/CodeGenProperties.h"
#include "dawn/IIR/Interval.h"
#include "dawn/Support/StringUtil.h"
#include <stack>
#include <unordered_map>

namespace dawn {

namespace iir {
class StencilInstantiation;
class StencilFunctionInstantiation;
}

namespace codegen {
namespace cuda {

/// @brief ASTVisitor to generate C++ naive code for the stencil and stencil function bodies
/// @ingroup cuda
class ASTStencilBody : public ASTCodeGenCXX {
protected:
  const iir::StencilInstantiation* instantiation_;
  RangeToString offsetPrinter_;

  ///
  /// @brief produces a string of (i,j,k) accesses for the C++ generated naive code,
  /// from an array of offseted accesses
  ///
  std::array<std::string, 3> ijkfyOffset(const std::array<int, 3>& offsets,
                                         std::string accessName) {
    int n = -1;
    std::array<std::string, 3> res;
    std::transform(offsets.begin(), offsets.end(), res.begin(), [&](int const& off) {
      ++n;
      std::array<std::string, 3> indices{"istride", "jstride", "kstride"};

      return off ? (indices[n] + "*" + std::to_string(off)) : "";
    });
    return res;
  }

public:
  using Base = ASTCodeGenCXX;

  /// @brief constructor
  ASTStencilBody(const iir::StencilInstantiation* stencilInstantiation,
                 StencilContext stencilContext);

  virtual ~ASTStencilBody();

  /// @name Statement implementation
  /// @{
  virtual void visit(const std::shared_ptr<ReturnStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<StencilCallDeclStmt>& stmt) override;
  virtual void visit(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) override;
  /// @}

  /// @name Expression implementation
  /// @{
  virtual void visit(const std::shared_ptr<FunCallExpr>& expr) override;
  virtual void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override;
  virtual void visit(const std::shared_ptr<StencilFunArgExpr>& expr) override;
  virtual void visit(const std::shared_ptr<VarAccessExpr>& expr) override;
  virtual void visit(const std::shared_ptr<FieldAccessExpr>& expr) override;
  /// @}

  /// @brief Mapping of VarDeclStmt and Var/FieldAccessExpr to their name
  std::string getName(const std::shared_ptr<Expr>& expr) const override;
  std::string getName(const std::shared_ptr<Stmt>& stmt) const override;
};

} // namespace cuda
} // namespace codegen
} // namespace dawn

#endif
