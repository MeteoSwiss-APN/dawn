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

#pragma once

#include <map>
#include <string>
#include <vector>

namespace dawn {
namespace codegen {

/// @brief Result of the code generation process
/// @ingroup codegen
class TranslationUnit {
  std::string filename_;                        ///< File of the translation unit
  std::vector<std::string> ppDefines_;          ///< Preprocessor defines
  std::string globals_;                         ///< Code for globals struct
  std::map<std::string, std::string> stencils_; ///< Code for each stencil mapped by name

public:
  using const_iterator = std::map<std::string, std::string>::const_iterator;

  TranslationUnit(const TranslationUnit&) = default;
  TranslationUnit(TranslationUnit&&) = default;

  /// @brief Construct the TranslationUnit by consuming the input arguments
  TranslationUnit(std::string filename, std::vector<std::string>&& ppDefines,
                  std::map<std::string, std::string>&& stencils, std::string&& globals);

  /// @brief Get filename
  const std::string& getFilename() const { return filename_; }

  /// @brief Get a list of preprocessor defines
  const std::vector<std::string>& getPPDefines() const { return ppDefines_; }

  /// @brief Get the map of the generated stencils (name/code pair)
  const std::map<std::string, std::string>& getStencils() const { return stencils_; }

  /// @brief Get the code for the globals struct
  const std::string& getGlobals() const { return globals_; }
};

} // namespace codegen
} // namespace dawn
