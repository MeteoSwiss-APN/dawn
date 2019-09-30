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

#ifndef DAWN_IIR_STATEMENTACCESSESPAIR_H
#define DAWN_IIR_STATEMENTACCESSESPAIR_H

#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/AccessToNameMapper.h"
#include "dawn/IIR/Accesses.h"
#include "dawn/IIR/BlockStatements.h"
#include "dawn/IIR/IIRNode.h"
#include "dawn/Support/Json.h"
#include <memory>
#include <optional>
#include <vector>
namespace dawn {
namespace iir {

class DoMethod;
class StencilMetaInformation;

/// @brief Statement with corresponding Accesses
///
/// If the statement is a block-statement, the sub-statements will be stored in blockStatements.
/// @ingroup optimizer
class StatementAccessesPair : public IIRNode<DoMethod, StatementAccessesPair, void> {

  std::shared_ptr<iir::Stmt> statement_;

  // In case of a non function call stmt, the accesses are stored in callerAccesses_, while
  // calleeAccesses_ will be nullptr

  // Accesses of the statement. If the statement is part of a stencil-function, this will store the
  // caller accesses. The caller access will have the initial offset added (e.g if a stencil
  // function is called with `avg(u(i+1))` the initial offset of `u` is `[1, 0, 0]`).
  std::shared_ptr<Accesses> callerAccesses_;

  // If the statement is part of a stencil-function, this will store the callee accesses i.e the
  // accesses without the initial offset of the call
  std::shared_ptr<Accesses> calleeAccesses_;

  // If the statement is a block statement, this will contain the sub-statements of the block. Note
  // that the acceses in this case are the *accumulated* accesses of all sub-statements.
  BlockStatements blockStatements_;

public:
  static constexpr const char* name = "StatementAccessesPair";

  inline virtual void updateFromChildren() override {}

  explicit StatementAccessesPair(const std::shared_ptr<iir::Stmt>& statement);

  StatementAccessesPair(StatementAccessesPair&&) = default;

  /// @brief clone the statement accesses pair, returning a smart ptr
  std::unique_ptr<StatementAccessesPair> clone() const;

  /// @brief Get/Set the statement
  std::shared_ptr<iir::Stmt> getStatement() const;
  void setStatement(const std::shared_ptr<iir::Stmt>& statement);

  /// @brief Get/Set the accesses
  std::shared_ptr<Accesses> getAccesses() const;
  void setAccesses(const std::shared_ptr<Accesses>& accesses);

  /// @brief Get/Set the caller accesses (alias for `getAccesses`)
  std::shared_ptr<Accesses> getCallerAccesses() const;
  void setCallerAccesses(const std::shared_ptr<Accesses>& accesses);

  /// @brief Get/Set the callee accesses (only set for statements inside stencil-functions)
  std::shared_ptr<Accesses> getCalleeAccesses() const;
  void setCalleeAccesses(const std::shared_ptr<Accesses>& accesses);
  bool hasCalleeAccesses();

  /// @brief Get the blockStatements
  const std::vector<std::unique_ptr<StatementAccessesPair>>& getBlockStatements() const;
  bool hasBlockStatements() const;

  /// @brief insert a new statemenent accesses pair as a block statement
  void insertBlockStatement(std::unique_ptr<StatementAccessesPair>&& stmt);

  std::optional<Extents> computeMaximumExtents(const int accessID) const;

  /// @brief Convert the StatementAccessesPair of a stencil or stencil-function to string
  /// @{
  std::string toString(const StencilMetaInformation* metadata, std::size_t initialIndent = 0) const;
  std::string toString(const StencilFunctionInstantiation* stencilFunc,
                       std::size_t initialIndent = 0) const;
  /// @}

  json::json jsonDump(const StencilMetaInformation& metadata) const;
  json::json print(const StencilMetaInformation& metadata,
                   const AccessToNameMapper& accessToNameMapper,
                   const std::unordered_map<int, Extents>& accesses) const;
};

} // namespace iir
} // namespace dawn

#endif
