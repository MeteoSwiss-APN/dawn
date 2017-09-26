//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GSL_COMPILER_DIAGNOSTICSENGINE_H
#define GSL_COMPILER_DIAGNOSTICSENGINE_H

#include "gsl/Compiler/DiagnosticsMessage.h"
#include "gsl/Compiler/DiagnosticsQueue.h"
#include "gsl/Support/NonCopyable.h"

namespace gsl {

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

} // namespace gsl

#endif
