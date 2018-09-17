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

#ifndef DAWN_IIR_BLOCKSTATEMENTS_H
#define DAWN_IIR_BLOCKSTATEMENTS_H

#include <memory>
#include <vector>

namespace dawn {
namespace iir {

class StatementAccessesPair;

class BlockStatements {

  // If the statement is a block statement, this will contain the sub-statements of the block. Note
  // that the acceses in this case are the *accumulated* accesses of all sub-statements.
  std::vector<std::unique_ptr<StatementAccessesPair>> blockStatements_;

public:
  BlockStatements(BlockStatements&&) = default;
  BlockStatements() = default;

  BlockStatements& operator=(BlockStatements&&) = default;
  BlockStatements clone() const;

  /// @brief Get the blockStatements
  const std::vector<std::unique_ptr<StatementAccessesPair>>& getBlockStatements() const;
  //  std::vector<std::unique_ptr<StatementAccessesPair>>& getBlockStatements();
  bool hasBlockStatements() const;
  /// @}

  /// @brief insert a new statement accesses pair
  void insert(std::unique_ptr<StatementAccessesPair>&& stmt);
};

} // namespace iir
} // namespace dawn

#endif
