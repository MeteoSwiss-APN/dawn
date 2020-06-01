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

#include "dawn/Optimizer/PassDataLocalityMetric.h"
#include "dawn/IIR/AST.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Logger.h"
#include "dawn/Support/StringUtil.h"
#include <deque>
#include <sstream>
#include <stack>
#include <unordered_map>
#include <unordered_set>

namespace dawn {

namespace {

class ReadWriteCounter : public iir::ASTVisitorForwarding {
  const iir::StencilMetaInformation& metadata_;
  OptimizerContext& context_;

  std::size_t numReads_, numWrites_;

  /// Current MultiStage
  const iir::MultiStage& multiStage_;

  /// Fields of the MultiStage
  std::unordered_map<int, iir::Field> fields_;

  /// Fields which are considered to be loaded into a register
  std::unordered_set<int> register_;

  /// Fields in the texture Cache as well as the associated k-level
  ///
  /// E.g if we access u(k-1) we store the k-1 level in the cache and a subsequent access to u(k)
  /// cannot be supplied from the texture cache
  std::deque<std::pair<int, int>> textureCache_;

  /// Fields loaded in K-Cache
  std::unordered_set<int> kCacheLoaded_;

  /// Current stencil function call
  std::stack<std::shared_ptr<iir::StencilFunctionInstantiation>> stencilFunCalls_;

  /// Map of ID's to their respective number of reads / writes
  std::unordered_map<int, ReadWriteAccumulator> individualReadWrites_;

public:
  ReadWriteCounter(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                   OptimizerContext& context, const iir::MultiStage& multiStage)
      : metadata_(instantiation->getMetaData()), context_(context), numReads_(0), numWrites_(0),
        multiStage_(multiStage), fields_(multiStage_.getFields()) {}

  std::size_t getNumReads() const { return numReads_; }
  std::size_t getNumWrites() const { return numWrites_; }

  void updateTextureCache(int AccessID, int kOffset) {
    if(textureCache_.size() < context_.getHardwareConfiguration().TexCacheMaxFields)
      textureCache_.emplace_front(AccessID, kOffset);
    else {
      auto it = std::find_if(
          textureCache_.begin(), textureCache_.end(),
          [&](const std::pair<int, int>& p) { return p.first == AccessID && p.second == kOffset; });

      // Update the Cache according to LRU
      if(it != textureCache_.end())
        textureCache_.erase(it);
      else
        textureCache_.pop_back();

      textureCache_.emplace_front(AccessID, kOffset);
    }
  }

  bool isTextureCached(int AccessID, int kOffset) {
    return std::find_if(textureCache_.begin(), textureCache_.end(),
                        [&](const std::pair<int, int>& p) {
                          return p.first == AccessID && p.second == kOffset;
                        }) != textureCache_.end();
  }

  std::string getNameFromAccessID(int AccessID) {
    return stencilFunCalls_.empty() ? metadata_.getFieldNameFromAccessID(AccessID)
                                    : stencilFunCalls_.top()->getFieldNameFromAccessID(AccessID);
  }

  std::shared_ptr<iir::StencilFunctionInstantiation>
  getStencilFunctionInstantiation(const std::shared_ptr<iir::StencilFunCallExpr>& expr) {
    return stencilFunCalls_.empty() ? metadata_.getStencilFunctionInstantiation(expr)
                                    : stencilFunCalls_.top()->getStencilFunctionInstantiation(expr);
  }

  ast::Offsets getOffset(const std::shared_ptr<iir::FieldAccessExpr>& field) {
    return stencilFunCalls_.empty()
               ? field->getOffset()
               : stencilFunCalls_.top()->evalOffsetOfFieldAccessExpr(field, true);
  };

  void updateKCache(int AccessID) {
    if(kCacheLoaded_.count(AccessID) || !multiStage_.isCached(AccessID))
      return;

    auto it =
        std::find_if(multiStage_.getCaches().begin(), multiStage_.getCaches().end(),
                     [&](const std::pair<int, iir::Cache>& p) { return p.first == AccessID; });
    DAWN_ASSERT(it != multiStage_.getCaches().end());
    const iir::Cache& cache = it->second;

    if(cache.getType() == iir::Cache::CacheType::K) {
      if(cache.getIOPolicy() == iir::Cache::IOPolicy::fill ||
         cache.getIOPolicy() == iir::Cache::IOPolicy::fill_and_flush) {
        numReads_++;
        individualReadWrites_[AccessID].numReads++;
      }
      if(cache.getIOPolicy() == iir::Cache::IOPolicy::flush ||
         cache.getIOPolicy() == iir::Cache::IOPolicy::fill_and_flush) {
        numWrites_++;
        individualReadWrites_[AccessID].numWrites++;
      }
    }
    kCacheLoaded_.insert(AccessID);
  }

  void processWriteAccess(const std::shared_ptr<iir::FieldAccessExpr>& field) {
    int AccessID = iir::getAccessID(field);

    // Is field stored in cache?
    if(!multiStage_.getCaches().count(AccessID)) {
      numWrites_++;
      individualReadWrites_[AccessID].numWrites++;

      // The written value is stored in a register
      register_.insert(AccessID);

    } else {
      updateKCache(AccessID);
    }
  }

  void processReadAccess(const std::shared_ptr<iir::FieldAccessExpr>& fieldExpr) {
    int AccessID = iir::getAccessID(fieldExpr);
    int kOffset = fieldExpr->getOffset().verticalOffset();

    auto it = fields_.find(AccessID);
    DAWN_ASSERT(it != fields_.end());
    iir::Field& field = it->second;

    if(field.getIntend() == iir::Field::IntendKind::Input) {
      if(!register_.count(AccessID)) {

        // Cache the center access
        if(getOffset(fieldExpr).isZero())
          register_.insert(AccessID);

        // Check if the field is either cached or stored in the texture cache
        if(!(multiStage_.isCached(AccessID) || isTextureCached(AccessID, kOffset))) {
          numReads_++;
          individualReadWrites_[AccessID].numReads++;
        } else {
          updateKCache(AccessID);
        }
      }
    } else {
      if(!multiStage_.isCached(AccessID)) {

        // Check if the center is stored in a register
        if(!register_.count(AccessID) || !getOffset(fieldExpr).isZero()) {
          numReads_++;
          individualReadWrites_[AccessID].numReads++;
        }

      } else {
        updateKCache(AccessID);
      }
    }

    if(multiStage_.isCached(AccessID) && field.getIntend() == iir::Field::IntendKind::Input)
      updateTextureCache(AccessID, kOffset);
  }

  void visit(const std::shared_ptr<iir::AssignmentExpr>& expr) override {
    // LHS is a write and maybe a read if we have an expression like `a += 5`
    bool readAndWrite = expr->getOp() == "+=" || expr->getOp() == "-=" || expr->getOp() == "/=" ||
                        expr->getOp() == "*=" || expr->getOp() == "|=" || expr->getOp() == "&=";

    if(isa<iir::FieldAccessExpr>(expr->getLeft().get())) {
      auto field = std::static_pointer_cast<iir::FieldAccessExpr>(expr->getLeft());
      processWriteAccess(field);
      if(readAndWrite)
        processReadAccess(field);
    }

    // RHS are read accesses
    expr->getRight()->accept(*this);
  }

  void visit(const std::shared_ptr<iir::StencilFunCallExpr>& expr) override {
    stencilFunCalls_.push(getStencilFunctionInstantiation(expr));
    stencilFunCalls_.top()->getAST()->accept(*this);
    stencilFunCalls_.pop();
  }

  void visit(const std::shared_ptr<iir::FieldAccessExpr>& expr) override {
    processReadAccess(expr);
  }
  const std::unordered_map<int, ReadWriteAccumulator>& getIndividualReadWrites() const {
    return individualReadWrites_;
  }
};

/// @brief Approximate an upperbound of the required reads and writes.
///
/// Every non-cached input field yiels has at most 1 main-memory access, input-output at most 2, and
/// every output at most 1 store.
DAWN_ATTRIBUTE_UNUSED static std::pair<int, int>
computeReadWriteAccessesLowerBound(iir::StencilInstantiation* instantiation,
                                   const iir::MultiStage& multiStage) {
  std::size_t numReads = 0, numWrites = 0;
  std::unordered_map<int, iir::Field> fields = multiStage.getFields();

  for(const auto& AccessIDFieldPair : fields) {
    int AccessID = AccessIDFieldPair.first;
    const iir::Field& field = AccessIDFieldPair.second;

    // The only fields we don't count are the once which are cached an *not* filled or flushed i.e
    // have a local fill policy (note that IJ caches are always local)
    if(multiStage.isCached(AccessID)) {
      const iir::Cache& cache = multiStage.getCaches().find(AccessID)->second;

      if(cache.getType() == iir::Cache::CacheType::K) {
        if(cache.getIOPolicy() == iir::Cache::IOPolicy::fill ||
           cache.getIOPolicy() == iir::Cache::IOPolicy::fill_and_flush) {
          numReads += 1;
        }
        if(cache.getIOPolicy() == iir::Cache::IOPolicy::flush ||
           cache.getIOPolicy() == iir::Cache::IOPolicy::fill_and_flush) {
          numWrites += 1;
        }
      }

    } else {
      switch(field.getIntend()) {
      case iir::Field::IntendKind::Output:
        numWrites += 1;
        break;
      case iir::Field::IntendKind::InputOutput:
        numReads += 1;
        numWrites += 1;
        break;
      case iir::Field::IntendKind::Input:
        numReads += 1;
        break;
      }
    }
  }

  return std::make_pair(numReads, numWrites);
}

} // anonymous namespace

/// @brief Approximate the reads and writes individually for each ID
std::unordered_map<int, ReadWriteAccumulator> computeReadWriteAccessesMetricPerAccessID(
    const std::shared_ptr<iir::StencilInstantiation>& instantiation, OptimizerContext& context,
    const iir::MultiStage& multiStage) {
  ReadWriteCounter readWriteCounter(instantiation, context, multiStage);

  for(const auto& stmt : iterateIIROverStmt(multiStage)) {
    stmt->accept(readWriteCounter);
  }

  return readWriteCounter.getIndividualReadWrites();
}

/// @brief Approximate the reads and writes accoding to our data locality metric
std::pair<int, int>
computeReadWriteAccessesMetric(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                               OptimizerContext& context, const iir::MultiStage& multiStage) {
  ReadWriteCounter readWriteCounter(instantiation, context, multiStage);

  for(const auto& stmt : iterateIIROverStmt(multiStage)) {
    stmt->accept(readWriteCounter);
  }

  return std::make_pair(readWriteCounter.getNumReads(), readWriteCounter.getNumWrites());
}

PassDataLocalityMetric::PassDataLocalityMetric(OptimizerContext& context)
    : Pass(context, "PassDataLocalityMetric") {}

bool PassDataLocalityMetric::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {

  std::stringstream ss;
  if(context_.getOptions().ReportDataLocalityMetric) {
    const std::string title = " DataLocality - " + stencilInstantiation->getName() + " ";
    const int paddingLength = std::max(int(TERMINAL_CHAR_WIDTH - title.size()), 0);
    ss << std::string((paddingLength) / 2, '-') << title
       << std::string((paddingLength + 1) / 2, '-') << "\n";

    std::size_t perStencilNumReads = 0, perStencilNumWrites = 0;

    int stencilIdx = 0;
    for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
      const iir::Stencil& stencil = *stencilPtr;

      ss << "Stencil " << stencilIdx << ":\n";

      int multiStageIdx = 0;
      for(const auto& multiStagePtr : stencil.getChildren()) {
        const iir::MultiStage& multiStage = *multiStagePtr;

        ss << "  MultiStage " << multiStageIdx << ":\n";

        auto readAndWrite =
            computeReadWriteAccessesMetric(stencilInstantiation, context_, multiStage);

        std::size_t numReads = readAndWrite.first, numWrites = readAndWrite.second;

        ss << format("    %-20s %15i\n", "Reads", numReads);
        ss << format("    %-20s %15i\n", "Writes", numWrites);

        DAWN_LOG(INFO) << ss.str();

        perStencilNumReads += numReads;
        perStencilNumWrites += numWrites;
        multiStageIdx++;
      }

      stencilIdx++;
    }

    DAWN_LOG(INFO) << "Reads: " << perStencilNumReads << ", Writes: " << perStencilNumWrites;
  }

  return true;
}

} // namespace dawn
