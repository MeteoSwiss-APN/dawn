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

#include "dawn/Optimizer/PassSetCaches.h"
#include "dawn/IIR/Cache.h"
#include "dawn/IIR/IntervalAlgorithms.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Support/Logger.h"
#include "dawn/Support/Unreachable.h"
#include <optional>
#include <set>
#include <vector>

namespace dawn {

namespace {

enum class FirstAccessKind { Read, Write, Mixed };

/// @brief properties of a cache candidate
struct CacheCandidate {
  iir::Cache::IOPolicy policy_;
  std::optional<iir::Cache::window> window_;
  iir::Interval interval_;
};

/// @brief Combine the policies from the first MultiStage (`MS1Policy`) and the immediately
/// following (`MS2Policy`)
///
/// MS1Policy has to be one of: {local, fill, bpfill}
/// MS2Policy has to be one of: {local, flush, epflush}
///
///                                       MS2
///
///                    ||    local    ||     fill      ||   bpfill     ||
///       =========++=================++===============++==============++
///        local       ||    local    |      flush     |    epflush     |
///       =============++-------------+----------------+----------------+
///  MS1   fill (ro)   ||    fill     |      fill      |      fill      |
///       =============++-------------+----------------+----------------+
///        fill (w)    ||    fill     + fill_and_flush + fill_and_flush +
///       =============++-------------+----------------+----------------+
///        bpfill (ro) ||   bpfill    |       fill     |      fill      |
///       =============++-------------+----------------+----------------+
///        bpfill (w ) ||   bpfill    | fill_and_flush | fill_and_flush |
///       =============++-------------+----------------+----------------+
///
CacheCandidate combinePolicy(CacheCandidate const& MS1Policy, iir::Field::IntendKind fieldIntend,
                             CacheCandidate const& MS2Policy) {
  if(MS1Policy.policy_ == iir::Cache::IOPolicy::local) {
    if(MS2Policy.policy_ == iir::Cache::IOPolicy::fill)
      return CacheCandidate{iir::Cache::IOPolicy::flush, std::make_optional(iir::Cache::window{}),
                            MS1Policy.interval_};
    if(MS2Policy.policy_ == iir::Cache::IOPolicy::bpfill) {
      DAWN_ASSERT(MS2Policy.window_);
      auto const& window = *(MS2Policy.window_);
      return CacheCandidate{iir::Cache::IOPolicy::epflush,
                            std::make_optional(iir::Cache::window{-window.m_p, -window.m_m}),
                            MS1Policy.interval_};
    }
    if(MS2Policy.policy_ == iir::Cache::IOPolicy::local)
      return MS2Policy;
    dawn_unreachable("Not valid policy");
  }
  if(MS1Policy.policy_ == iir::Cache::IOPolicy::fill ||
     MS1Policy.policy_ == iir::Cache::IOPolicy::bpfill) {
    if(MS2Policy.policy_ == iir::Cache::IOPolicy::fill ||
       MS2Policy.policy_ == iir::Cache::IOPolicy::bpfill) {
      if(fieldIntend == iir::Field::IntendKind::Input)
        return CacheCandidate{iir::Cache::IOPolicy::fill, MS1Policy.window_, MS1Policy.interval_};
      else
        return CacheCandidate{iir::Cache::IOPolicy::fill_and_flush, MS1Policy.window_,
                              MS1Policy.interval_};
    }
    if(MS2Policy.policy_ == iir::Cache::IOPolicy::local)
      return MS1Policy;
    dawn_unreachable("Not valid policy");
  }
  dawn_unreachable("invalid policy combination");
}

/// computes a cache candidate of a field for a multistage
CacheCandidate computeCacheCandidateForMS(iir::Field const& field, bool isTemporaryField,
                                          iir::MultiStage const& MS) {

  if(field.getIntend() == iir::Field::IntendKind::Input) {
    std::optional<iir::Interval> interval =
        MS.computeEnclosingAccessInterval(field.getAccessID(), true);
    // make sure the access interval has the same boundaries as from any interval of the mss
    interval->merge(field.getInterval());
    DAWN_ASSERT(interval);

    return CacheCandidate{iir::Cache::IOPolicy::fill, std::optional<iir::Cache::window>(),
                          *interval};
  }
  if(field.getIntend() == iir::Field::IntendKind::Output) {
    return CacheCandidate{iir::Cache::IOPolicy::local, std::optional<iir::Cache::window>(),
                          field.getInterval()};
  }

  if(field.getIntend() == iir::Field::IntendKind::InputOutput) {

    std::optional<iir::Interval> interval =
        MS.computeEnclosingAccessInterval(field.getAccessID(), true);

    DAWN_ASSERT(interval);

    DAWN_ASSERT(interval->contains(field.getInterval()));

    iir::MultiInterval multiInterval = MS.computeReadAccessInterval(field.getAccessID());
    if(multiInterval.empty())
      return CacheCandidate{iir::Cache::IOPolicy::local, std::optional<iir::Cache::window>(),
                            field.getInterval()};

    if(multiInterval.numPartitions() > 1 ||
       multiInterval.getIntervals()[0].contains(field.getInterval()))
      return CacheCandidate{iir::Cache::IOPolicy::fill, std::optional<iir::Cache::window>(),
                            *interval};

    iir::Interval const& readInterval = multiInterval.getIntervals()[0];

    iir::Cache::window window =
        computeWindowOffset(MS.getLoopOrder(), readInterval, field.getInterval());

    if(((MS.getLoopOrder() == iir::LoopOrderKind::Forward) && window.m_m <= 0 && window.m_p <= 0) ||
       ((MS.getLoopOrder() == iir::LoopOrderKind::Backward) && window.m_m >= 0 &&
        window.m_p >= 0)) {
      return CacheCandidate{
          iir::Cache::IOPolicy::bpfill,
          std::make_optional(iir::Cache::window{
              ((MS.getLoopOrder() == iir::LoopOrderKind::Forward) ? window.m_m : 0),
              ((MS.getLoopOrder() == iir::LoopOrderKind::Forward) ? 0 : window.m_p)}),
          *interval};
    }

    return CacheCandidate{iir::Cache::IOPolicy::fill, std::optional<iir::Cache::window>(),
                          *interval};
  }
  dawn_unreachable("Policy of Field not found");
}

} // anonymous namespace

PassSetCaches::PassSetCaches(OptimizerContext& context) : Pass(context, "PassSetCaches") {}

bool PassSetCaches::run(const std::shared_ptr<iir::StencilInstantiation>& instantiation) {
  const auto& metadata = instantiation->getMetaData();

  for(const auto& stencilPtr : instantiation->getStencils()) {
    iir::Stencil& stencil = *stencilPtr;

    // Set IJ-Caches
    int msIdx = 0;
    for(const auto& multiStagePtr : stencil.getChildren()) {

      iir::MultiStage& MS = *(multiStagePtr);

      std::set<int> outputFields;

      auto isOutput = [](const iir::Field& field) {
        return field.getIntend() == iir::Field::IntendKind::Output ||
               field.getIntend() == iir::Field::IntendKind::InputOutput;
      };

      for(const auto& stage : MS.getChildren()) {
        for(const auto& fieldPair : stage->getFields()) {
          const iir::Field& field = fieldPair.second;
          const int accessID = field.getAccessID();
          const iir::Field& msField = MS.getField(accessID);

          // Field is already cached, skip
          if(MS.isCached(accessID)) {
            continue;
          }

          // Field has vertical extents, can't be ij-cached
          if(!msField.getExtents().isVerticalPointwise()) {
            continue;
          }

          // Currently we only cache temporaries!
          if(!metadata.isAccessType(iir::FieldAccessType::StencilTemporary, accessID)) {
            continue;
          }

          // Cache the field
          if(field.getIntend() == iir::Field::IntendKind::Input && outputFields.count(accessID) &&
             !field.getExtents().isHorizontalPointwise()) {

            iir::Cache& cache =
                MS.setCache(iir::Cache::CacheType::IJ, iir::Cache::IOPolicy::local, accessID);

            DAWN_LOG(INFO) << instantiation->getName() << ": MS" << msIdx << ": "
                           << instantiation->getOriginalNameFromAccessID(accessID) << ":"
                           << cache.getTypeAsString() << ":" << cache.getIOPolicyAsString();
          }

          if(isOutput(field))
            outputFields.insert(accessID);
        }
      }
      msIdx++;
    }

    // Set K-Caches
    if(!context_.getOptions().DisableKCaches ||
       stencil.getStencilAttributes().has(sir::Attr::Kind::UseKCaches)) {

      std::set<int> mssProcessedFields;
      for(int MSIndex = 0; MSIndex < stencil.getChildren().size(); ++MSIndex) {
        iir::MultiStage& ms = *stencil.getMultiStageFromMultiStageIndex(MSIndex);
        const auto& fields = ms.getFields();
        for(const auto& AccessIDFieldPair : fields) {
          const iir::Field& field = AccessIDFieldPair.second;
          bool mssProcessedField = mssProcessedFields.count(field.getAccessID());
          if(!mssProcessedField)
            mssProcessedFields.emplace(field.getAccessID());

          // Field is already cached, skip
          if(ms.isCached(field.getAccessID()))
            continue;

          // Field has horizontal extents, can't be k-cached
          if(!field.getExtents().isHorizontalPointwise())
            continue;
          // we dont know how to cache fields with out of center writes
          if(field.getWriteExtents() && !field.getWriteExtents()->isPointwise())
            continue;

          if(!metadata.isAccessType(iir::FieldAccessType::StencilTemporary, field.getAccessID()) &&
             (field.getIntend() == iir::Field::IntendKind::Output ||
              (field.getIntend() == iir::Field::IntendKind::Input &&
               field.getExtents().isPointwise())))
            continue;

          // If the Field about to be cached is in a stage with a global index
          // it can in fact only be chached if all subsequent stages have compatible iteration
          // spaces

          // find first stage in which the field is used
          int stageIter = 0;
          for(const auto& stage : ms.getChildren()) {
            if(stage->getFields().count(field.getAccessID()) && stage->hasIterationSpace()) {
              break;
            }
            stageIter++;
          }

          // check if subsequent uses are compatible
          auto firstUsageWithGlobalIndex = ms.getChildren().begin();
          std::advance(firstUsageWithGlobalIndex, stageIter);
          bool incompatibleIterationSpaces = false;
          for(auto it = firstUsageWithGlobalIndex; it != ms.getChildren().end(); it++) {
            bool stageUsesField = it->get()->getFields().count(field.getAccessID());
            if(stageUsesField &&
               !firstUsageWithGlobalIndex->get()->iterationSpaceCompatible(*it->get())) {
              incompatibleIterationSpaces = true;
              break;
            }
          }
          if(incompatibleIterationSpaces) {
            continue;
          }

          // Determine if we need to fill the cache by analyzing the current multi-stage
          CacheCandidate cacheCandidate = computeCacheCandidateForMS(
              field,
              metadata.isAccessType(iir::FieldAccessType::StencilTemporary, field.getAccessID()),
              ms);

          if(!metadata.isAccessType(iir::FieldAccessType::StencilTemporary, field.getAccessID()) &&
             field.getIntend() != iir::Field::IntendKind::Input) {

            cacheCandidate = combinePolicy(cacheCandidate, field.getIntend(),
                                           CacheCandidate{iir::Cache::IOPolicy::fill,
                                                          std::optional<iir::Cache::window>(),
                                                          /* FirstAccessKind::Read, */
                                                          field.getInterval()});
          } else {

            for(int MSIndex2 = MSIndex + 1; MSIndex2 < stencil.getChildren().size(); MSIndex2++) {
              iir::MultiStage& nextMS = *stencil.getMultiStageFromMultiStageIndex(MSIndex2);
              const auto& nextMSfields = nextMS.getFields();

              if(!nextMSfields.count(field.getAccessID()))
                continue;

              const iir::Field& fieldInNextMS = nextMSfields.find(field.getAccessID())->second;

              CacheCandidate policyMS2 = computeCacheCandidateForMS(
                  fieldInNextMS,
                  metadata.isAccessType(iir::FieldAccessType::StencilTemporary,
                                        fieldInNextMS.getAccessID()),
                  nextMS);
              // if the interval of the two cache candidate do not overlap, there is no data
              // dependency, therefore we skip it
              if(!cacheCandidate.interval_.overlaps(policyMS2.interval_))
                continue;

              cacheCandidate = combinePolicy(cacheCandidate, field.getIntend(), policyMS2);
              break;
            }
          }

          iir::Interval interval = field.getInterval();
          auto interval_ = ms.computeEnclosingAccessInterval(field.getAccessID(), true);
          DAWN_ASSERT(interval_);
          auto enclosingAccessedInterval = *interval_;

          // Set the cache
          iir::Cache& cache =
              ms.setCache(iir::Cache::CacheType::K, cacheCandidate.policy_, field.getAccessID(),
                          interval, enclosingAccessedInterval, cacheCandidate.window_);

          DAWN_LOG(INFO) << instantiation->getName() << ": MS" << MSIndex << ": "
                         << instantiation->getOriginalNameFromAccessID(field.getAccessID()) << ":"
                         << cache.getTypeAsString() << ":" << cache.getIOPolicyAsString()
                         << (cache.getWindow() ? (std::string(":") + cache.getWindow()->toString())
                                               : "");
        }
      }
    }
  }

  return true;
}

} // namespace dawn
