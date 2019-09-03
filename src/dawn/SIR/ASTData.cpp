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

#include "dawn/SIR/ASTData.h"
#include "dawn/SIR/SIR.h"
#include <memory>

namespace dawn {
namespace sir {

SIRASTData::VerticalRegionDeclStmt::VerticalRegionDeclStmt(
    const std::shared_ptr<VerticalRegion>& verticalRegion)
    : verticalRegion_(verticalRegion) {}

SIRASTData::VerticalRegionDeclStmt::VerticalRegionDeclStmt(const VerticalRegionDeclStmt& stmtData)
    : StmtData(), verticalRegion_(std::make_shared<VerticalRegion>(*stmtData.verticalRegion_)) {}

SIRASTData::VerticalRegionDeclStmt& SIRASTData::VerticalRegionDeclStmt::
operator=(VerticalRegionDeclStmt const& stmtData) {
  return *this = VerticalRegionDeclStmt(stmtData);
}

} // namespace sir
} // namespace dawn
