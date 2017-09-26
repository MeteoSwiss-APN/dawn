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

#ifndef DAWN_COMPILER_DIAGNOSTICQUEUE_H
#define DAWN_COMPILER_DIAGNOSTICQUEUE_H

#include "dawn/Compiler/DiagnosticsMessage.h"
#include "dawn/Support/NonCopyable.h"
#include <memory>
#include <vector>

namespace dawn {

/// @brief Queue of diagnostic messages
/// @ingroup compiler
class DiagnosticsQueue : public NonCopyable {
  std::vector<std::unique_ptr<DiagnosticsMessage>> queue_;
  unsigned numErrors_;
  unsigned numWarnings_;

public:
  using const_iterator = std::vector<std::unique_ptr<DiagnosticsMessage>>::const_iterator;

  DiagnosticsQueue();

  /// @brief Push a diagnostic message to the end of the queue
  void push_back(const DiagnosticsMessage& msg);
  void push_back(DiagnosticsMessage&& msg);

  /// @brief Number of errors occured
  unsigned getNumErros() { return numErrors_; }

  /// @brief Number of warnings occured
  unsigned getNumWarnings() { return numWarnings_; }

  /// @brief Clear the diagnostic queue
  void clear();

  /// @brief Check if there are any errors
  bool hasErrors() const { return (numErrors_ != 0); }

  /// @brief Check if there are any warnings
  bool hasWarnings() const { return (numWarnings_ != 0); }

  /// @brief Get the vector of diagnostic messages in the order they were inserted
  const std::vector<std::unique_ptr<DiagnosticsMessage>>& queue() const { return queue_; }

  /// @name Iterator interface
  /// @{
  const_iterator begin() const { return queue_.cbegin(); }
  const_iterator end() const { return queue_.cend(); }
  /// @}
};

} // namespace dawn

#endif
