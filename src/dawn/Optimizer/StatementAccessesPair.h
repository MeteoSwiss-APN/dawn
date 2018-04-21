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

#ifndef DAWN_OPTIMIZER_STATEMENTACCESSESPAIR_H
#define DAWN_OPTIMIZER_STATEMENTACCESSESPAIR_H

#include "dawn/Optimizer/Accesses.h"
#include "dawn/SIR/Statement.h"
#include <boost/optional.hpp>
#include <memory>
#include <vector>

namespace dawn {

/// @brief Statement with corresponding Accesses
///
/// If the statement is a block-statement, the sub-statements will be stored in `children`.
/// @ingroup optimizer
class StatementAccessesPair {
  std::shared_ptr<Statement> statement_;

  // Accesses of the statement. If the statement is part of a stencil-function, this will store the
  // caller accesses. The caller access will have the initial offset added (e.g if a stencil
  // function is called with `avg(u(i+1))` the initial offset of `u` is `[1, 0, 0]`).
  std::shared_ptr<Accesses> callerAccesses_;

  // If the statement is part of a stencil-function, this will store the callee accesses i.e the
  // accesses without the initial offset of the call
  std::shared_ptr<Accesses> calleeAccesses_;

  // If the statement is a block statement, this will contain the sub-statements of the block. Note
  // that the acceses in this case are the *accumulated* accesses of all sub-statements.
  std::vector<std::shared_ptr<StatementAccessesPair>> children_;

public:
  explicit StatementAccessesPair(const std::shared_ptr<Statement>& statement);

  /// @brief Get/Set the statement
  std::shared_ptr<Statement> getStatement() const;
  void setStatement(const std::shared_ptr<Statement>& statement);

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

  /// @brief Get the children
  const std::vector<std::shared_ptr<StatementAccessesPair>>& getChildren() const;
  std::vector<std::shared_ptr<StatementAccessesPair>>& getChildren();
  bool hasChildren() const;

  boost::optional<Extents> computeMaximumExtents(const int accessID) const;

  /// @brief Convert the StatementAccessesPair of a stencil or stencil-function to string
  /// @{
  std::string toString(const StencilInstantiation* instantiation,
                       std::size_t initialIndent = 0) const;
  std::string toString(const StencilFunctionInstantiation* stencilFunc,
                       std::size_t initialIndent = 0) const;
  /// @}
};

} // namespace dawn

#endif
