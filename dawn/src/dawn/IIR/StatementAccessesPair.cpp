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

#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/ASTStringifier.h"
#include "dawn/IIR/AccessToNameMapper.h"
#include "dawn/IIR/Accesses.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include "dawn/Support/Printing.h"
#include <sstream>

namespace dawn {
namespace iir {

namespace {

template <class InstantiationType>
static std::string toStringImpl(const StatementAccessesPair* pair,
                                const InstantiationType* instantiation, std::size_t initialIndent) {
  std::stringstream ss;

  std::string initialIndentStr = std::string(initialIndent, ' ');
  std::size_t curIndent = initialIndent + DAWN_PRINT_INDENT;
  std::string curIndentStr = std::string(curIndent, ' ');

  ss << initialIndentStr << "[\n" << curIndentStr << "Statement:\n";
  ss << "\e[1m" << ASTStringifier::toString(pair->getStatement(), curIndent + DAWN_PRINT_INDENT)
     << "\e[0m\n";

  if(const auto& accesses = pair->getStatement()->getData<iir::IIRStmtData>().CallerAccesses) {
    ss << curIndentStr << "CallerAccesses:\n";
    ss << accesses->toString(instantiation, curIndent + DAWN_PRINT_INDENT) << "\n";
  }

  if(const auto& accesses = pair->getStatement()->getData<iir::IIRStmtData>().CalleeAccesses) {
    ss << curIndentStr << "CalleeAccesses:\n";
    ss << accesses->toString(instantiation, curIndent + DAWN_PRINT_INDENT) << "\n";
  }

  if(!pair->getBlockStatements().empty()) {
    ss << curIndentStr << "BlockStatements:\n";
    for(auto& child : pair->getBlockStatements())
      ss << child->toString(instantiation, curIndent);
  }
  ss << initialIndentStr << "]\n";

  return ss.str();
}

} // anonymous namespace

StatementAccessesPair::StatementAccessesPair(const std::shared_ptr<iir::Stmt>& statement)
    : statement_(statement) {}

std::unique_ptr<StatementAccessesPair> StatementAccessesPair::clone() const {
  auto cloneSAP = std::make_unique<StatementAccessesPair>(statement_->clone());

  cloneSAP->blockStatements_ = blockStatements_.clone();

  cloneSAP->cloneChildrenFrom(*this);

  return cloneSAP;
}

std::shared_ptr<iir::Stmt> StatementAccessesPair::getStatement() const { return statement_; }

void StatementAccessesPair::setStatement(const std::shared_ptr<iir::Stmt>& statement) {
  statement_ = statement;
}

const std::vector<std::unique_ptr<StatementAccessesPair>>&
StatementAccessesPair::getBlockStatements() const {
  return blockStatements_.getBlockStatements();
}

void StatementAccessesPair::insertBlockStatement(std::unique_ptr<StatementAccessesPair>&& stmt) {
  blockStatements_.insert(std::move(stmt));
}

bool StatementAccessesPair::hasBlockStatements() const {
  return blockStatements_.hasBlockStatements();
}

json::json StatementAccessesPair::print(const StencilMetaInformation& metadata,
                                        const AccessToNameMapper& accessToNameMapper,
                                        const std::unordered_map<int, Extents>& accesses) const {
  json::json node;
  for(const auto& accessPair : accesses) {
    json::json accessNode;
    int accessID = accessPair.first;
    std::string accessName = "unknown";
    if(accessToNameMapper.hasAccessID(accessID)) {
      accessName = accessToNameMapper.getNameFromAccessID(accessID);
    }
    if(metadata.isAccessType(iir::FieldAccessType::FAT_Literal, accessID)) {
      continue;
    }
    accessNode["access_id"] = accessID;
    accessNode["name"] = accessName;
    std::stringstream ss;
    ss << accessPair.second;
    accessNode["extents"] = ss.str();
    node.push_back(accessNode);
  }
  return node;
}

json::json StatementAccessesPair::jsonDump(const StencilMetaInformation& metadata) const {
  json::json node;
  node["stmt"] = ASTStringifier::toString(getStatement(), 0);

  AccessToNameMapper accessToNameMapper(metadata);
  getStatement()->accept(accessToNameMapper);

  node["write_accesses"] =
      print(metadata, accessToNameMapper,
            getStatement()->getData<iir::IIRStmtData>().CallerAccesses->getWriteAccesses());
  node["read_accesses"] =
      print(metadata, accessToNameMapper,
            getStatement()->getData<iir::IIRStmtData>().CallerAccesses->getReadAccesses());
  return node;
}

std::string StatementAccessesPair::toString(const StencilMetaInformation* metadata,
                                            std::size_t initialIndent) const {
  return toStringImpl(this, metadata, initialIndent);
}

std::string StatementAccessesPair::toString(const StencilFunctionInstantiation* stencilFunc,
                                            std::size_t initialIndent) const {
  return toStringImpl(this, stencilFunc, initialIndent);
}

} // namespace iir
} // namespace dawn
