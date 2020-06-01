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

#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/Assert.h"
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

namespace dawn {
namespace codegen {
// @brief context of a stencil body
// (pure stencil or a stencil function)
enum class StencilContext { SC_Stencil = 0, SC_StencilFunction };

/// @brief struct to store properties of a stencil for code generation
/// @ingroup codegen
struct StencilProperties {
  std::unordered_map<std::string, std::string> paramNameToType_;
  const int id_;
  const std::string name_;
  StencilProperties(const int id, const std::string name) : id_(id), name_(name) {}
};

/// @brief global metadata and properties of stencils needed for code generation
/// @ingroup codegen
class CodeGenProperties {
  struct Impl {
    std::unordered_map<std::string, std::shared_ptr<StencilProperties>> stencilProps_;
    std::unordered_map<size_t, std::string> stencilIDToName_;
  };

  // map of parameter name to its type
  std::unordered_map<std::string, std::string> paramNameToType_;
  // set of fields that are used as boundary conditions
  std::set<std::string> paramBC_;
  // map of parameter position to its name
  std::unordered_map<size_t, std::string> paramPositionIdxToName_;

  // set of fields allocate by the stencil wrapper
  std::set<std::string> allocatedFields_;

  // array stencil properties. The elements of the array corresponds to
  // SC_Stencil and SC_StencilFunction
  std::array<Impl, 2> stencilContextProperties_;

public:
  /// @brief mark a specific parameter as boundary condition param. Note the parameter should have
  /// been inserted
  void setParamBC(std::string name);

  /// @brief true if the parameter is also a BC field
  bool isParamBC(std::string name) const;

  /// @brief insert a parameter in the mapping data structures
  void insertParam(const size_t paramPosition, std::string paramName, std::string paramType);

  /// @brief insert the name of an allocated field
  void insertAllocateField(std::string name);

  /// @brief get the set of allocated fields
  std::set<std::string> getAllocatedFields() const;

  /// @brief true if stencil has allocated fields
  bool hasAllocatedFields() const;

  /// @brief get the type associate to parameter with name paramName
  std::string getParamType(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                           const std::string paramName) const;

  /// @brief stencil properties map getter
  std::unordered_map<std::string, std::shared_ptr<StencilProperties>>&
  stencilProperties(StencilContext context);

  /// @brief insert a new stencil properties
  std::shared_ptr<StencilProperties> insertStencil(StencilContext context, const int id,
                                                   const std::string name);

  /// @brief stencil properties getter
  std::shared_ptr<StencilProperties> getStencilProperties(StencilContext context,
                                                          const std::string name);

  const std::shared_ptr<StencilProperties>& getStencilProperties(StencilContext context,
                                                                 const std::string name) const;

  /// @brief stencil properties getter
  std::shared_ptr<StencilProperties> getStencilProperties(StencilContext context, const int id);

  const std::shared_ptr<StencilProperties>& getStencilProperties(StencilContext context,
                                                                 const int id) const;

  /// @brief get all the stencil properties
  const std::unordered_map<std::string, std::shared_ptr<StencilProperties>>&
  getAllStencilProperties(StencilContext context) const;

  /// @brief stencil name getter
  std::string getStencilName(StencilContext context, const size_t id) const;

  /// @brief get parameter
  const std::unordered_map<std::string, std::string>& getParameterNameToType() const;

  /// @brief get the parameter type
  std::string getParamType(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                           const iir::Stencil::FieldInfo& field) const;

private:
  std::shared_ptr<StencilProperties> insertStencil(Impl& impl, const size_t id,
                                                   const std::string name);
};
} // namespace codegen
} // namespace dawn
