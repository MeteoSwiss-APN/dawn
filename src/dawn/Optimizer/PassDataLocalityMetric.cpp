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
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/AST.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/StringUtil.h"
#include <deque>
#include <stack>
#include <unordered_map>
#include <unordered_set>

namespace dawn {

namespace {

class ReadWriteCounter : public ASTVisitorForwarding {
  const std::shared_ptr<iir::StencilInstantiation>& instantiation_;

  std::size_t numReads_, numWrites_;

  /// Current MultiStage
  const iir::MultiStage& multiStage_;

  /// Fields of the MultiStage
  std::unordered_map<int, Field> fields_;

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
                   const iir::MultiStage& multiStage)
      : instantiation_(instantiation), numReads_(0), numWrites_(0), multiStage_(multiStage),
        fields_(multiStage_.getFields()) {}

  std::size_t getNumReads() const { return numReads_; }
  std::size_t getNumWrites() const { return numWrites_; }

  void updateTextureCache(int AccessID, int kOffset) {
    if(textureCache_.size() <
       instantiation_->getOptimizerContext()->getHardwareConfiguration().TexCacheMaxFields)
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

  int getAccessIDFromExpr(const std::shared_ptr<Expr>& expr) {
    return stencilFunCalls_.empty() ? instantiation_->getAccessIDFromExpr(expr)
                                    : stencilFunCalls_.top()->getAccessIDFromExpr(expr);
  }

  std::string getNameFromAccessID(int AccessID) {
    return stencilFunCalls_.empty() ? instantiation_->getNameFromAccessID(AccessID)
                                    : stencilFunCalls_.top()->getNameFromAccessID(AccessID);
  }

  std::shared_ptr<iir::StencilFunctionInstantiation>
  getStencilFunctionInstantiation(const std::shared_ptr<StencilFunCallExpr>& expr) {
    return stencilFunCalls_.empty() ? instantiation_->getStencilFunctionInstantiation(expr)
                                    : stencilFunCalls_.top()->getStencilFunctionInstantiation(expr);
  }

  Array3i getOffset(const std::shared_ptr<FieldAccessExpr>& field) {
    return stencilFunCalls_.empty()
               ? field->getOffset()
               : stencilFunCalls_.top()->evalOffsetOfFieldAccessExpr(field, true);
  };

  void updateKCache(int AccessID) {
    if(kCacheLoaded_.count(AccessID) || !multiStage_.isCached(AccessID))
      return;

    auto it = std::find_if(multiStage_.getCaches().begin(), multiStage_.getCaches().end(),
                           [&](const std::pair<int, Cache>& p) { return p.first == AccessID; });
    DAWN_ASSERT(it != multiStage_.getCaches().end());
    const Cache& cache = it->second;

    if(cache.getCacheType() == Cache::K) {
      if(cache.getCacheIOPolicy() == Cache::fill ||
         cache.getCacheIOPolicy() == Cache::fill_and_flush) {
        numReads_++;
        individualReadWrites_[AccessID].numReads++;
      }
      if(cache.getCacheIOPolicy() == Cache::flush ||
         cache.getCacheIOPolicy() == Cache::fill_and_flush) {
        numWrites_++;
        individualReadWrites_[AccessID].numWrites++;
      }
    }
    kCacheLoaded_.insert(AccessID);
  }

  void processWriteAccess(const std::shared_ptr<FieldAccessExpr>& field) {
    int AccessID = getAccessIDFromExpr(field);

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

  void processReadAccess(const std::shared_ptr<FieldAccessExpr>& fieldExpr) {
    int AccessID = getAccessIDFromExpr(fieldExpr);
    int kOffset = fieldExpr->getOffset()[2];

    auto it = fields_.find(AccessID);
    DAWN_ASSERT(it != fields_.end());
    Field& field = it->second;

    if(field.getIntend() == Field::IK_Input) {
      if(!register_.count(AccessID)) {

        // Cache the center access
        if(getOffset(fieldExpr) == Array3i{{0, 0, 0}})
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
        if(!(register_.count(AccessID) && getOffset(fieldExpr) == Array3i{{0, 0, 0}})) {
          numReads_++;
          individualReadWrites_[AccessID].numReads++;
        }

      } else {
        updateKCache(AccessID);
      }
    }

    if(multiStage_.isCached(AccessID) && field.getIntend() == Field::IK_Input)
      updateTextureCache(AccessID, kOffset);
  }

  void visit(const std::shared_ptr<AssignmentExpr>& expr) override {
    // LHS is a write and maybe a read if we have an expression like `a += 5`
    bool readAndWrite = StringRef(expr->getOp()) == "+=" || StringRef(expr->getOp()) == "-=" ||
                        StringRef(expr->getOp()) == "/=" || StringRef(expr->getOp()) == "*=" ||
                        StringRef(expr->getOp()) == "|=" || StringRef(expr->getOp()) == "&=";

    if(isa<FieldAccessExpr>(expr->getLeft().get())) {
      auto field = std::static_pointer_cast<FieldAccessExpr>(expr->getLeft());
      processWriteAccess(field);
      if(readAndWrite)
        processReadAccess(field);
    }

    // RHS are read accesses
    expr->getRight()->accept(*this);
  }

  void visit(const std::shared_ptr<StencilFunCallExpr>& expr) override {
    stencilFunCalls_.push(getStencilFunctionInstantiation(expr));
    stencilFunCalls_.top()->getAST()->accept(*this);
    stencilFunCalls_.pop();
  }

  void visit(const std::shared_ptr<FieldAccessExpr>& expr) override { processReadAccess(expr); }
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
  std::unordered_map<int, Field> fields = multiStage.getFields();

  for(const auto& AccessIDFieldPair : fields) {
    int AccessID = AccessIDFieldPair.first;
    const Field& field = AccessIDFieldPair.second;

    // The only fields we don't count are the once which are cached an *not* filled or flushed i.e
    // have a local fill policy (note that IJ caches are always local)
    if(multiStage.isCached(AccessID)) {
      const Cache& cache = multiStage.getCaches().find(AccessID)->second;

      if(cache.getCacheType() == Cache::K) {
        if(cache.getCacheIOPolicy() == Cache::fill ||
           cache.getCacheIOPolicy() == Cache::fill_and_flush) {
          numReads += 1;
        }
        if(cache.getCacheIOPolicy() == Cache::flush ||
           cache.getCacheIOPolicy() == Cache::fill_and_flush) {
          numWrites += 1;
        }
      }

    } else {
      switch(field.getIntend()) {
      case Field::IK_Output:
        numWrites += 1;
        break;
      case Field::IK_InputOutput:
        numReads += 1;
        numWrites += 1;
        break;
      case Field::IK_Input:
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
    const std::shared_ptr<iir::StencilInstantiation>& instantiation,
    const iir::MultiStage& multiStage) {
  ReadWriteCounter readWriteCounter(instantiation, multiStage);

  for(const auto& stage : multiStage.getStages())
    for(const auto& doMethod : stage->getDoMethods())
      for(const auto& statementAccessesPair : doMethod->getStatementAccessesPairs()) {
        statementAccessesPair->getStatement()->ASTStmt->accept(readWriteCounter);
      }

  return readWriteCounter.getIndividualReadWrites();
}

/// @brief Approximate the reads and writes accoding to our data locality metric
std::pair<int, int>
computeReadWriteAccessesMetric(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                               const iir::MultiStage& multiStage) {
  ReadWriteCounter readWriteCounter(instantiation, multiStage);

  for(const auto& stage : multiStage.getStages())
    for(const auto& doMethod : stage->getDoMethods())
      for(const auto& statementAccessesPair : doMethod->getStatementAccessesPairs()) {
        statementAccessesPair->getStatement()->ASTStmt->accept(readWriteCounter);
      }

  return std::make_pair(readWriteCounter.getNumReads(), readWriteCounter.getNumWrites());
}

PassDataLocalityMetric::PassDataLocalityMetric() : Pass("PassDataLocalityMetric") {}

bool PassDataLocalityMetric::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  OptimizerContext* context = stencilInstantiation->getOptimizerContext();

  if(context->getOptions().ReportDataLocalityMetric) {
    std::string title = " DataLocality - " + stencilInstantiation->getName() + " ";
    std::cout << std::string((51 - title.size()) / 2, '-') << title
              << std::string((51 - title.size() + 1) / 2, '-') << "\n";

    std::size_t perStencilNumReads = 0, perStencilNumWrites = 0;

    int stencilIdx = 0;
    for(const auto& stencilPtr : stencilInstantiation->getStencils()) {
      const iir::Stencil& stencil = *stencilPtr;

      std::cout << "Stencil " << stencilIdx << ":\n";

      int multiStageIdx = 0;
      for(const auto& multiStagePtr : stencil.getMultiStages()) {
        const iir::MultiStage& multiStage = *multiStagePtr;

        std::cout << "  MultiStage " << multiStageIdx << ":\n";

        auto readAndWrite = computeReadWriteAccessesMetric(stencilInstantiation, multiStage);

        //        auto readAndWrite =
        //            computeReadWriteAccessesLowerBound(stencilInstantiation, multiStage);

        std::size_t numReads = readAndWrite.first, numWrites = readAndWrite.second;

        std::cout << format("    %-20s %15i\n", "Reads", numReads);
        std::cout << format("    %-20s %15i\n", "Writes", numWrites);

        perStencilNumReads += numReads;
        perStencilNumWrites += numWrites;
        multiStageIdx++;
      }

      stencilIdx++;
    }

    std::cout << format("\n  %-22s %15s\n", "", std::string(15, '='));
    std::cout << format("  %-22s %15i\n", "Reads", perStencilNumReads);
    std::cout << format("  %-22s %15i\n", "Writes", perStencilNumWrites);
    std::cout << std::string(51, '-') << std::endl;
  }

  return true;
}

} // namespace dawn
