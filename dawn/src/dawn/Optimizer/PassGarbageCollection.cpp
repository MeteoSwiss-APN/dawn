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

#include "PassGarbageCollection.h"
#include "dawn/AST/LocationType.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTFwd.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/DependencyGraphStage.h"
#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/Stage.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include "dawn/Support/Exception.h"
#include "dawn/Support/Logger.h"
#include <deque>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>

namespace dawn {

// struct IIRNodeDeleter : public iir::ASTVisitorForwarding {};

bool PassGarbageCollection::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const Options& options) {
  return true;
}

} // namespace dawn
