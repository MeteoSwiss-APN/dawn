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

#ifndef DAWN_SUPPORT_DIAGNOSTICMESSAGE_H
#define DAWN_SUPPORT_DIAGNOSTICMESSAGE_H

#include "dawn/Support/SourceLocation.h"
#include <sstream>
#include <string>

namespace dawn {

enum class DiagnosticsKind { Note, Warning, Error };

/// @brief Representation of a diagnostics message
/// @ingroup compiler
class DiagnosticsMessage {
  DiagnosticsKind diag_;
  SourceLocation loc_;
  std::string filename_;
  std::string msg_;

public:
  /// @brief Initialize a diagnostics message
  ///
  /// Use DiagnosticsBuilder to assemble the message.
  ///
  /// @param diag      Kind of diagnostic
  /// @param loc       Location in the source
  /// @param filename  File where the diagnostic occurred
  /// @param msg       Message of the diagnostic
  DiagnosticsMessage(DiagnosticsKind diag, const SourceLocation& loc, const std::string& filename,
                     const std::string& msg)
      : diag_(diag), loc_(loc), filename_(filename), msg_(msg) {}

  DiagnosticsMessage(const DiagnosticsMessage&) = default;
  DiagnosticsMessage(DiagnosticsMessage&&) = default;

  DiagnosticsMessage& operator=(const DiagnosticsMessage&) = default;
  DiagnosticsMessage& operator=(DiagnosticsMessage&&) = default;

  /// @brief Get the kind of diagnostic
  DiagnosticsKind getDiagKind() const { return diag_; }
  void setDiagKind(DiagnosticsKind diag) { diag_ = diag; }

  /// @brief Get the source location
  const SourceLocation& getSourceLocation() const { return loc_; }
  void setSourceLocation(const SourceLocation& loc) { loc_ = loc; }

  /// @brief Get the filename
  const std::string& getFilename() const { return filename_; }
  void setFilename(const std::string& filename) { filename_ = filename; }

  /// @brief Get the message of the diagnostic
  const std::string& getMessage() const { return msg_; }
  void setMessage(const std::string& msg) { msg_ = msg; }
};

/// @brief Simplify construction of DiagnosticsMessages
/// @ingroup compiler
class DiagnosticsBuilder {
  DiagnosticsKind diag_;
  SourceLocation loc_;
  std::string msg_;

public:
  DiagnosticsBuilder(DiagnosticsKind diag, SourceLocation loc = SourceLocation())
      : diag_(diag), loc_(loc) {}

  DiagnosticsBuilder(const DiagnosticsBuilder&) = default;
  DiagnosticsBuilder(DiagnosticsBuilder&&) = default;

  /// @brief Stream content to the message
  template <class T>
  DiagnosticsBuilder& operator<<(T&& value) {
    // We could store the string stream as a member... but gcc-4.9 can't generate the default move
    // constructor
    std::ostringstream ss;
    ss << value;
    msg_ += ss.str();
    return *this;
  }

  /// @brief Assemble the final diagnostics message
  DiagnosticsMessage getMessage(const std::string& filename) const {
    return DiagnosticsMessage(diag_, loc_, filename, msg_);
  }
};

} // namespace dawn

#endif
