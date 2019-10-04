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

#ifndef DAWN_SUPPORT_DIAGNOSTICSENGINE_H
#define DAWN_SUPPORT_DIAGNOSTICSENGINE_H

#include "dawn/Support/DiagnosticsMessage.h"
#include "dawn/Support/DiagnosticsQueue.h"
#include "dawn/Support/NonCopyable.h"

namespace dawn {

/// @brief Concrete class used to report problems and issues
/// @ingroup compiler
class DiagnosticsEngine : NonCopyable {
  std::string filename_;
  DiagnosticsQueue queue_;

public:
  /// @brief Clear the diagnostic queue
  void clear() { queue_.clear(); }

  /// @brief Check if there are any diagnostics
  bool hasDiags() const { return hasErrors() || hasWarnings(); }

  /// @brief Check if there are any warnings
  bool hasWarnings() const { return queue_.hasWarnings(); }

  /// @brief Check if there are any errors
  bool hasErrors() const { return queue_.hasErrors(); }

  /// @brief Get the diagnostics queue
  const DiagnosticsQueue& getQueue() const { return queue_; }

  /// @brief Report a diagnostic
  void report(const DiagnosticsBuilder& diagBuilder);
  void report(const DiagnosticsMessage& diag);
  void report(DiagnosticsMessage&& diag);

  /// @brief Set the name of the file currently being processed
  void setFilename(const std::string& filename) { filename_ = filename; }
};

} // namespace dawn

#endif
