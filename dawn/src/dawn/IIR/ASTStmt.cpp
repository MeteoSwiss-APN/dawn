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

#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/Extents.h"
#include <memory>

namespace dawn {
namespace iir {

//===------------------------------------------------------------------------------------------===//
//     IIRStmtData
//===------------------------------------------------------------------------------------------===//

IIRStmtData::IIRStmtData(const IIRStmtData& other)
    : StackTrace(other.StackTrace), CallerAccesses(other.CallerAccesses),
      CalleeAccesses(other.CalleeAccesses) {}

bool IIRStmtData::operator==(const IIRStmtData& rhs) const {
  return StackTrace == rhs.StackTrace && CallerAccesses == rhs.CallerAccesses &&
         CalleeAccesses == rhs.CalleeAccesses;
}
bool IIRStmtData::operator!=(const IIRStmtData& rhs) const { return !(*this == rhs); }

std::unique_ptr<ast::StmtData> IIRStmtData::clone() const {
  return std::make_unique<IIRStmtData>(*this);
}

bool IIRStmtData::equals(ast::StmtData const* other) const {
  return other && getDataType() == other->getDataType() &&
         *this == dynamic_cast<IIRStmtData const&>(*other);
}

//===------------------------------------------------------------------------------------------===//
//     VarDeclStmtData
//===------------------------------------------------------------------------------------------===//

VarDeclStmtData::VarDeclStmtData(const VarDeclStmtData& other)
    : IIRStmtData(other), AccessID(other.AccessID) {}

bool VarDeclStmtData::operator==(const VarDeclStmtData& rhs) const {
  return IIRStmtData::operator==(rhs) && AccessID == rhs.AccessID;
}
bool VarDeclStmtData::operator!=(const VarDeclStmtData& rhs) const { return !(*this == rhs); }

std::unique_ptr<ast::StmtData> VarDeclStmtData::clone() const {
  return std::make_unique<VarDeclStmtData>(*this);
}

bool VarDeclStmtData::equals(ast::StmtData const* other) const {
  VarDeclStmtData const* varDeclStmtDataOther;
  return (varDeclStmtDataOther = dynamic_cast<VarDeclStmtData const*>(other)) &&
         *this == *varDeclStmtDataOther;
}

//===------------------------------------------------------------------------------------------===//
//     computeMaximumExtents
//===------------------------------------------------------------------------------------------===//

std::optional<Extents> computeMaximumExtents(Stmt& stmt, const int accessID) {
  std::optional<Extents> extents;

  const auto& callerAccesses = stmt.getData<IIRStmtData>().CallerAccesses;

  if(callerAccesses->hasReadAccess(accessID))
    extents = std::make_optional(callerAccesses->getReadAccess(accessID));

  if(callerAccesses->hasWriteAccess(accessID)) {
    if(extents)
      extents->merge(callerAccesses->getWriteAccess(accessID));
    else
      extents = std::make_optional(callerAccesses->getWriteAccess(accessID));
  }

  for(auto const& child : stmt.getChildren())
    if(auto childExtent = computeMaximumExtents(*child, accessID)) {
      if(extents)
        extents->merge(*childExtent);
      else
        extents = childExtent;
    }

  return extents;
}

int getAccessID(const std::shared_ptr<VarDeclStmt>& stmt) {
  return *stmt->getData<VarDeclStmtData>().AccessID;
}

} // namespace iir
} // namespace dawn
