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

IIRStmtData::IIRStmtData(const IIRStmtData& other) {
  StackTrace = other.StackTrace ? std::make_optional(*other.StackTrace) : other.StackTrace;
  CallerAccesses =
      other.CallerAccesses ? std::make_optional(*other.CallerAccesses) : other.CallerAccesses;
  CalleeAccesses =
      other.CalleeAccesses ? std::make_optional(*other.CalleeAccesses) : other.CalleeAccesses;
}

bool IIRStmtData::operator==(const IIRStmtData& rhs) {
  return StackTrace == rhs.StackTrace && CallerAccesses == rhs.CallerAccesses &&
         CalleeAccesses == rhs.CalleeAccesses;
}
bool IIRStmtData::operator!=(const IIRStmtData& rhs) { return !(*this == rhs); }

std::unique_ptr<ast::StmtData> IIRStmtData::clone() const {
  return std::make_unique<IIRStmtData>(*this);
}

//===------------------------------------------------------------------------------------------===//
//     VarDeclStmtData
//===------------------------------------------------------------------------------------------===//

VarDeclStmtData::VarDeclStmtData(const VarDeclStmtData& other) : IIRStmtData(other) {
  AccessID = other.AccessID ? std::make_optional(*other.AccessID) : other.AccessID;
}

bool VarDeclStmtData::operator==(const VarDeclStmtData& rhs) {
  return IIRStmtData::operator==(rhs) && AccessID == rhs.AccessID;
}
bool VarDeclStmtData::operator!=(const VarDeclStmtData& rhs) { return !(*this == rhs); }

std::unique_ptr<ast::StmtData> VarDeclStmtData::clone() const {
  return std::make_unique<VarDeclStmtData>(*this);
}

//===------------------------------------------------------------------------------------------===//
//     computeMaximumExtents
//===------------------------------------------------------------------------------------------===//

std::optional<Extents> computeMaximumExtents(Stmt& stmt, const int accessID) {
  std::optional<Extents> extents;

  const auto& callerAccesses = stmt.getData<IIRStmtData>().CallerAccesses;

  if(callerAccesses->hasReadAccess(accessID) || callerAccesses->hasWriteAccess(accessID)) {
    extents = std::optional<Extents>();

    if(callerAccesses->hasReadAccess(accessID)) {
      if(!extents)
        extents = std::make_optional(callerAccesses->getReadAccess(accessID));
      else
        extents->merge(callerAccesses->getReadAccess(accessID));
    }
    if(callerAccesses->hasWriteAccess(accessID)) {
      if(!extents)
        extents = std::make_optional(callerAccesses->getWriteAccess(accessID));
      else
        extents->merge(callerAccesses->getWriteAccess(accessID));
    }
  }

  for(auto const& child : stmt.getChildren()) {
    auto childExtent = computeMaximumExtents(*child, accessID);
    if(!childExtent)
      continue;
    if(extents)
      extents->merge(*childExtent);
    else
      extents = childExtent;
  }

  return extents;
}

} // namespace iir
} // namespace dawn
