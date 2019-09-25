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
#include <boost/optional.hpp>
#include <memory>
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

  /// @brief Get the blockStatements
  const std::vector<std::unique_ptr<StatementAccessesPair>>& getBlockStatements() const;
  bool hasBlockStatements() const;

  /// @brief insert a new statemenent accesses pair as a block statement
  void insertBlockStatement(std::unique_ptr<StatementAccessesPair>&& stmt);

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
